from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from shared.metrics import CategoricalMetric

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


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

    print(f"Using device {dev}")
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
    loss_fn,
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
        tuple: (computed_metrics, running_loss, confusion_matrix)
    """
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

            if len(metrics) > 0:
                preds = preds.cpu().numpy()
                labels = labels.cpu().numpy()
            for metric in metrics:
                metric.update(preds, labels)

        computed_metrics = [metric.compute().item() for metric in metrics]
        for metric in metrics:
            metric.reset()

    cm = confusion_matrix(all_labels, all_preds)

    return computed_metrics, running_loss, cm


def train(
    model: nn.Module,
    epochs: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn,
    optimizer: torch.optim.Optimizer,
    metrics: list[CategoricalMetric] = [],
    metric_names: list[str] = [],
    patience: int = 5,
    from_epoch: int = 10,
    print_batch_info=True
):
    """
    Train the model with early stopping based on validation loss.

    Parameters:
        model (nn.Module): The model to train.
        epochs (int): Number of epochs to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        loss_fn: Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for updating the model parameters.
        metrics (List[CategoricalMetric]): List of metrics to compute during training. Default is an empty list.
        metric_names (List[str]): List of metric names corresponding to the metrics. Default is an empty list.
        patience (int): Number of epochs to wait for improvement in validation loss before stopping. Default is 5.
        from_epoch (int): Number of epochs after which to start early stopping. Default is 10.
        print_batch_info (bool): If True, print batch information during training. Default is True.



    Returns:
        A generator yielding the validation loss after each epoch.
        After the last epoch returns
        tuple: (best_val_loss, best_model_confusion_matrix, best_model_metrics)

    Example::
        while True:
            try:
                running_loss = next(train_gen)
            except StopIteration as res:
                best_val_loss, best_confusion_matrix, best_model_metrics = res.value
    """

    device = set_device()
    model.to(device)
    loss_fn.to(device)

    best_val_loss = float('inf')
    best_model_metrics = []
    best_model_weights = None
    best_model_confusion_matrix = None
    early_stopping_counter = 0

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        model.train()
        running_loss = 0.0

        for data in train_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            _, preds = torch.max(outputs.data, 1)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if len(metrics) > 0:
                preds = preds.cpu().numpy()
                labels = labels.cpu().numpy()
            for metric in metrics:
                metric.update(preds, labels)

            train_metrics = [metric.compute().item() for metric in metrics]

            if print_batch_info:
                print_stats(metric_names, train_metrics,
                            running_loss, None, None, replace_ln=True)

        computed_metrics = [metric.compute().item() for metric in metrics]
        for metric in metrics:
            metric.reset()

        val_metrics, val_loss, val_confusion_matrix = evaluate(
            model, val_loader, loss_fn, metrics, device)

        yield val_loss

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = model.state_dict()
            best_model_confusion_matrix = val_confusion_matrix
            best_model_metrics = computed_metrics
            early_stopping_counter = 0
        elif epoch >= from_epoch:
            early_stopping_counter += 1

        print_stats(
            metric_names,
            computed_metrics, running_loss,
            val_metrics, val_loss,
            replace_ln=False
        )

        if early_stopping_counter >= patience:
            break

    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    return best_val_loss, best_model_confusion_matrix, best_model_metrics
