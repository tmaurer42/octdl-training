from dataclasses import dataclass
import math
from collections import OrderedDict
import copy

import numpy as np
import flwr as fl
from flwr.common.logger import log
from flwr.common import NDArrays
from logging import ERROR
import torch
from torch import nn
from torch.utils.data import DataLoader

from shared.training import LossFnType, train, evaluate
from shared.data import OCTDLClass
from shared.model import ModelType, get_model_by_type
from shared.metrics import CategoricalMetric


def get_largest_parameter_value(model: torch.nn.Module) -> float:
    max_val = float('-inf')  # Initialize to negative infinity

    # Iterate over all parameters in the model
    for param in model.parameters():
        if param is not None:
            param_max = torch.max(torch.abs(param)).item()
            max_val = max(max_val, param_max)

    return max_val


@dataclass
class ClientConfig:
    device: torch.device
    model_type: ModelType
    transfer_learning: bool
    dropout: float
    epochs: int
    lr: float
    loss_fn_type: LossFnType
    metrics: list[type[CategoricalMetric]]


class FlClient(fl.client.NumPyClient):
    def __init__(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        classes: list[OCTDLClass],
        config: ClientConfig
    ) -> None:
        super().__init__()

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = config.device
        self.classes = classes
        self.epochs = config.epochs
        self.lr = config.lr
        self.loss_fn_type = config.loss_fn_type
        self.metrics = [
            m() for m in config.metrics
        ]

        self.model = get_model_by_type(
            config.model_type,
            config.transfer_learning,
            classes,
            config.dropout
        )

    def set_parameters(self, parameters):
        model_state_dict_keys = list(self.model.state_dict().keys())

        params_dict = zip(model_state_dict_keys, parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def compute_class_weights(self):
        # Step 1: Count the number of samples per class
        class_counts = {}
        for _, labels in self.train_loader:
            labels = labels.cpu().numpy()
            for label in labels:
                if label in class_counts:
                    class_counts[label] += 1
                else:
                    class_counts[label] = 1

        # Convert class_counts to a list where index represents class label
        num_classes = len(class_counts)
        counts = np.zeros(num_classes)
        for label, count in class_counts.items():
            counts[label] = count

        # Step 2: Compute the class weights
        total_samples = np.sum(counts)
        class_weights = total_samples / (num_classes * counts)

        # Step 3: Normalize the weights if needed (optional)
        # For example, normalize such that the weights sum to 1
        class_weights = class_weights / np.sum(class_weights)

        return torch.tensor(class_weights, dtype=torch.float)

    def get_loss_fn(self):
        if self.loss_fn_type == 'CrossEntropy':
            loss_fn = nn.CrossEntropyLoss()
        elif self.loss_fn_type == 'WeightedCrossEntropy':
            balancing_weights = self.compute_class_weights()
            loss_fn = nn.CrossEntropyLoss(
                weight=balancing_weights, label_smoothing=0.1)

        return loss_fn

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        loss_fn = self.get_loss_fn()
        optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        train_gen = train(
            model=self.model,
            epochs=self.epochs,
            train_loader=self.train_loader,
            loss_fn=loss_fn,
            optimizer=optim,
            device=self.device,
            print_batch_info=False,
            print_epoch_info=False,
            adapt_lr=(lambda b: self.lr * b/self.train_loader.batch_size)
        )
        for round in train_gen:
            if math.isnan(round.train_loss):
                log(ERROR, "Train loss is nan!")

        return self.get_parameters({}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        loss_fn = self.get_loss_fn()

        metrics, loss, _ = evaluate(
            model=self.model,
            data_loader=self.val_loader,
            loss_fn=loss_fn,
            metrics=self.metrics,
            device=self.device
        )

        metrics_dict = {}
        for i, metric_val in enumerate(metrics):
            name = self.metrics[i].name()
            metrics_dict[name] = metric_val

        if math.isnan(loss):
            largest_param = get_largest_parameter_value(self.model)
            log(
                ERROR,
                f"client loss is nan, valloader lenght: {len(self.val_loader)}, val data size: {len(self.val_loader.dataset)}"
                f"Max param size is {largest_param}"
            )

        return float(loss), len(self.val_loader.dataset), metrics_dict


class FedBuffClient(FlClient):
    def __init__(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        classes: list[OCTDLClass],
        config: ClientConfig
    ) -> None:
        super().__init__(train_loader, val_loader, classes, config)
        named_model_params = self.model.named_parameters()
        self.trainable_param_names = [
            name for name, param in named_model_params if param.requires_grad]

    def set_parameters(self, parameters: NDArrays):
        new_state_dict = OrderedDict()
        assert len(parameters) == len(self.trainable_param_names)
        trainable_param_index = 0
        for name, param in self.model.state_dict().items():
            if name in self.trainable_param_names:
                p = parameters[trainable_param_index]
                new_state_dict[name] = torch.tensor(p)
                trainable_param_index += 1
            else:
                new_state_dict[name] = param

        self.model.load_state_dict(new_state_dict, strict=True)

    def get_parameters(self, config):
        parameters = []
        for name, params in self.model.state_dict().items():
            if name in self.trainable_param_names:
                parameters.append(params.cpu().numpy())

        assert len(parameters) == len(self.trainable_param_names)

        return parameters

    def fit(self, parameters: NDArrays, config):
        """
        FedBuff-client algorithm line 4:
        Compute the parameter update, i.e. subtract the trained params from the received ones.
        """
        received_parameters = copy.deepcopy(parameters)
        new_parameters, num_examples, _ = super().fit(parameters, config)

        update = [received - new for received,
                  new in zip(received_parameters, new_parameters)]

        return update, num_examples, {'staleness': config['staleness']}


def generate_client_fn(
    train_loaders: list[DataLoader],
    val_loaders: list[DataLoader],
    classes: list[OCTDLClass],
    config: ClientConfig
):
    def client_fn(ctx: fl.common.Context):
        partition_id = ctx.node_config['partition-id']
        return FlClient(
            train_loader=train_loaders[int(partition_id)],
            val_loader=val_loaders[int(partition_id)],
            classes=classes,
            config=config
        ).to_client()

    # return the function to spawn client
    return client_fn


def generate_fedbuff_client_fn(
    train_loaders: list[DataLoader],
    val_loaders: list[DataLoader],
    classes: list[OCTDLClass],
    config: ClientConfig
):
    def client_fn(ctx: fl.common.Context):
        partition_id = ctx.node_config['partition-id']
        return FedBuffClient(
            train_loader=train_loaders[int(partition_id)],
            val_loader=val_loaders[int(partition_id)],
            classes=classes,
            config=config
        ).to_client()

    # return the function to spawn client
    return client_fn
