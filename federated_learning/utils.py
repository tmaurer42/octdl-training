from collections import OrderedDict
import os

import torch
import numpy as np
import flwr as fl
from flwr.server.server import Parameters
from flwr.common import parameters_to_ndarrays

from shared.metrics import CategoricalMetric


def get_avg_metrics_fn(metric_types: list[type[CategoricalMetric]]) -> fl.common.MetricsAggregationFn:
    def average_metrics(metrics):
        metrics_dict = {}
        for m in metric_types:
            metric_name = m.name()
            avg_val = np.mean([metric[metric_name] for _, metric in metrics])
            metrics_dict[metric_name] = avg_val

        return metrics_dict

    return average_metrics


def apply_parameters(params: Parameters, model: torch.nn.Module, trainable_params_only=False):
    params_ndarrays = parameters_to_ndarrays(params)
    if trainable_params_only:
        new_state_dict = OrderedDict()
        trainable_param_names = [
            name for name, param in model.named_parameters() if param.requires_grad]
        assert len(params_ndarrays) == len(trainable_param_names)
        trainable_param_index = 0
        for name, param in model.state_dict().items():
            if name in trainable_param_names:
                p = params_ndarrays[trainable_param_index]
                new_state_dict[name] = torch.tensor(p)
                trainable_param_index += 1
            else:
                new_state_dict[name] = param
    else:
        model_state_dict_keys = list(model.state_dict().keys())

        params_dict = zip(model_state_dict_keys, params_ndarrays)
        new_state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

    model.load_state_dict(new_state_dict, strict=True)
