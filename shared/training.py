from dataclasses import dataclass
from typing import Callable, Literal, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import numpy as np

from shared.metrics import CategoricalMetric, F1ScoreMacro

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


OptimizationMode = Literal["minimize_loss", "maximize_f1_macro"]
LossFnType = Literal["CrossEntropy", "WeightedCrossEntropy"]


@dataclass
class EarlyStopping:
    patience: int
    from_epoch: int
    optimization_mode: Optional[OptimizationMode] = 'minimize_loss'


@dataclass
class TrainEpochResult:
    train_loss: float
    val_metrics: dict[str, float]
    val_loss: float
    val_confusion_matrix: np.ndarray
    model_weights: dict[str, any]


def set_device():
    """
    Set the computing device to GPU (CUDA), 
    MPS (Mac M1-M3 GPU), or CPU based on availability.

    Returns:
        torch.device: The selected device.
    """
    if torch.cuda.is_available():
        dev = 'cuda:0'
    elif torch.backends.mps.is_available():
        if torch.backends.mps.is_built():
            dev = "mps"
        else:
            print("MPS not available because the current PyTorch install was not "
                  "built with MPS enabled.")
    else:
        dev = "cpu"
        
    device = torch.device(dev)

    return device


def print_stats(
    metric_names: list[str],
    metric_values: list[float],
    loss: float,
    val_metric_values: Optional[list[float]] = None,
    val_loss: Optional[float] = None,
    replace_ln: bool = False
):
    """
    Print training and validation statistics.

    Parameters:
        metric_names (list[str]): List of metric names.
        metric_values (list[float]): List of training metric values.
        loss (float): Training loss value.
        val_metric_values (Optional[list[float]]): List of validation metric values. Default is None.
        val_loss (Optional[float]): Validation loss value. Default is None.
        replace_ln (bool): If True, replace the last printed line. Default is False.
    """
    assert len(metric_names) == len(
        metric_values), "Metric names and values must have the same length."
    if val_metric_values is not None:
        assert len(metric_names) == len(
            val_metric_values), "Metric names and validation metric values must have the same length."

    train_stats = ", ".join([f"{name}: {value:0.4f}" for name, value in zip(
        metric_names, metric_values)]) + ", "
    train_stats += f"loss: {loss:0.4f}"
    message = train_stats

    if val_metric_values is not None and val_loss is not None:
        val_stats = ", ".join([f"val_{name}: {value:0.4f}" for name, value in zip(
            metric_names, val_metric_values)]) + ", "
        val_stats += f"val_loss: {val_loss:0.4f}"
        message += " || " + val_stats

    print(
        message,
        end='\r' if replace_ln else '\n',
        flush=replace_ln
    )


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.CrossEntropyLoss,
    metrics: list[CategoricalMetric],
    device: torch.device
):
    """
    Evaluate the model on the given data loader.

    Parameters:
        model (nn.Module): The model to evaluate.
        data_loader (DataLoader): DataLoader for the evaluation dataset.
        loss_fn: Loss function.
        metrics (List[CategoricalMetric]): List of metrics to compute.
        device (torch.device): Device to run the evaluation on.

    Returns:
        tuple: (computed_metrics, avg_loss, confusion_matrix)
    """
    model.to(device)
    loss_fn.to(device)
    model.eval()
    
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs.data, 1)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            preds = preds.cpu().numpy()
            labels = labels.cpu().numpy()
            for metric in metrics:
                metric.update(preds, labels)

    avg_loss = running_loss / len(data_loader)

    computed_metrics = [metric.compute() for metric in metrics]
    for metric in metrics:
        metric.reset()

    cm = confusion_matrix(all_labels, all_preds)

    return computed_metrics, avg_loss, cm


def train(
    model: nn.Module,
    epochs: int,
    train_loader: DataLoader,
    loss_fn,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    metrics: Optional[list[CategoricalMetric]] = [],
    val_loader: Optional[DataLoader] = None,
    early_stopping: Optional[EarlyStopping] = None,
    adapt_lr: Optional[Callable[[float], float]] = None,
    print_batch_info = False,
    print_epoch_info = True,
):
    """
    Train the model with early stopping based on validation loss.

    Parameters:
        model (nn.Module): The model to train.
        epochs (int): Number of epochs to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset. If val_loader is None, skip the evaluation step and therefore don't use early stopping.
        loss_fn: Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for updating the model parameters.
        metrics (List[CategoricalMetric]): List of metrics to compute during training. Default is an empty list.
        early_stopping (EarlyStopping): The early stopping strategy to use. Has no effect, if val_loader is None.
        print_batch_info (bool): If True, print batch information during training. Default is True.


    Returns:
        A generator yielding the TrainEpochResult each epoch. If validation is skipped, return None instead
    """
    model.to(device)
    loss_fn.to(device)

    metric_names = [metric.name() for metric in metrics]

    best_val_loss = float('inf')
    best_f1_macro = float('-inf')
    early_stopping_counter = 0

    for epoch in range(epochs):
        if print_epoch_info:
            print(f"Epoch {epoch + 1}")
        model.train()
        running_loss = 0.0
        all_preds = torch.tensor([], device=device)
        all_labels = torch.tensor([],  device=device)

        for i, data in enumerate(train_loader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            if adapt_lr is not None and len(data) != train_loader.batch_size:
                for group in optimizer.param_groups:
                    group['lr'] = adapt_lr(len(labels))

            optimizer.zero_grad()
            outputs = model(images)
            _, preds = torch.max(outputs.data, 1)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            all_preds = torch.cat((all_preds, preds), dim=0)
            all_labels = torch.cat((all_labels, labels), dim=0)

            batch_loss = loss.item()
            running_loss += batch_loss

            if print_batch_info:
                preds = preds.cpu().numpy()
                labels = labels.cpu().numpy()
                train_metrics = [metric.update(preds, labels) for metric in metrics]

                train_loss = running_loss / (i + 1)
                if print_batch_info:
                    print_stats(metric_names, train_metrics,
                                train_loss, None, None, replace_ln=True)
        
        train_loss = running_loss / len(train_loader)
        [metric.update(
            all_preds.cpu().tolist(),
            all_labels.cpu().tolist()
        ) for metric in metrics]

        train_metrics = [metric.compute() for metric in metrics]
        for metric in metrics:
            metric.reset()

        epoch_result, val_metrics, metrics_dict, val_loss, val_confusion_matrix = None, None, None, None, None
        if val_loader is not None:
            val_metrics, val_loss, val_confusion_matrix = evaluate(
                model, val_loader, loss_fn, metrics, device)

            metrics_dict = {name: val for name, val in zip(metric_names, val_metrics)}

        epoch_result = TrainEpochResult(train_loss,
            metrics_dict, val_loss, val_confusion_matrix, model.state_dict())
        
        yield epoch_result

        if print_epoch_info:
            print_stats(
                metric_names,
                train_metrics, train_loss,
                val_metrics, val_loss,
                replace_ln=False
            )

        if early_stopping is not None and val_loader is not None:
            if early_stopping.optimization_mode == 'maximize_f1_macro':
                val_f1_macro = val_metrics[F1ScoreMacro.name()]
                if val_f1_macro > best_f1_macro:
                    best_f1_macro = val_f1_macro
                    early_stopping_counter = 0
                elif epoch >= early_stopping.from_epoch:
                    early_stopping_counter += 1

            if early_stopping.optimization_mode == 'minimize_loss':
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    early_stopping_counter = 0
                elif epoch >= early_stopping.from_epoch:
                    early_stopping_counter += 1

            if early_stopping_counter >= early_stopping.patience:
                break

def train_optimized(
    model: nn.Module,
    epochs: int,
    train_loader: DataLoader,
    loss_fn,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    adapt_lr: Optional[Callable[[float], float]] = None,
):
    model.to(device)
    loss_fn.to(device)

    model.train()
    for _ in range(epochs):
        for data in train_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            if adapt_lr is not None and len(data) != train_loader.batch_size:
                for group in optimizer.param_groups:
                    group['lr'] = adapt_lr(len(data))

            for param in model.parameters():
                param.grad = None
            outputs = model(images)

            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

def eval_optimized(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.CrossEntropyLoss,
    metrics: list[CategoricalMetric],
    device: torch.device,
):
    model.to(device)
    loss_fn.to(device)
    
    running_loss = 0.0
    all_preds = torch.tensor([], device=device)
    all_labels = torch.tensor([],  device=device)

    model.eval()
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs.data, 1)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()

            all_preds = torch.cat((all_preds, preds), dim=0)
            all_labels = torch.cat((all_labels, labels), dim=0)

    avg_loss = running_loss / len(data_loader)

    all_preds = all_preds.cpu().numpy()
    all_labels = all_labels.cpu().numpy()

    for metric in metrics:
        metric.update(all_preds, all_labels)
    computed_metrics = [metric.compute() for metric in metrics]
    for metric in metrics:
        metric.reset()

    cm = confusion_matrix(all_labels, all_preds)

    return computed_metrics, avg_loss, cm