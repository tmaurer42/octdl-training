from typing import Optional

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchmetrics import Metric, Accuracy

from data import OCTDLClass, load_octdl_dataset


def set_device():
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
        metric_name: str,
        metric_value: float, 
        loss: float, 
        val_metric_value: Optional[float], 
        val_loss: Optional[float],
        replace_ln: bool = False
    ):
    train_stats = f"{metric_name}: {metric_value:0.4f}, loss: {loss:0.4f}" 
    message = train_stats

    if val_metric_value is not None and val_loss is not None:
        val_stats = f"val_{metric_name}: {val_metric_value:0.4f}, val_loss: {val_loss:0.4f}"
        message += " || " + val_stats

    print(
        message, 
        end='\r' if replace_ln else None,
        flush=replace_ln
    )


def evaluate(model: nn.Module, data_loader: DataLoader, metric: Metric, device: torch.device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs.data, 1)
            loss = loss_fn(outputs, labels)
            running_loss += loss
            metric.update(preds, labels)
        
        metric_value = metric.compute().item()
        metric.reset()

    return metric_value, running_loss



def train(
        model: nn.Module, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        loss_fn,
        optimizer: optim.Optimizer,
        metric: Metric,
        metric_name,
        epochs: int
    ):
    device = set_device()
    model.to(device)
    metric.to(device)

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
            metric_value = metric(preds, labels)

            print_stats(metric_name, metric_value.item(), running_loss, None, None, replace_ln=True)
        
        metric_value = metric.compute().item()
        metric.reset()

        val_metric_value, val_loss = evaluate(model, val_loader, metric, device)
        print_stats(
            metric_name, 
            metric_value, running_loss, 
            val_metric_value, val_loss, 
            replace_ln=False
        )


if __name__ == "__main__":
    classes = [OCTDLClass.AMD, OCTDLClass.NO]
    transfer_learning = False
    image_size = 224

    batch_size = 32
    learning_rate = 0.001
    epochs = 10

    mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

    val_test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Normalize(mean=mean, std=std)
    ])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.Normalize(mean=mean, std=std),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.GaussianBlur(kernel_size=(3, 5))
    ])

    train_ds, val_ds, test_ds = load_octdl_dataset(
        classes,
        train_transform,
        val_test_transform
    )

    assert(isinstance(train_ds.__getitem__(0)[1], int))
    print(len(train_ds.data))
    print(len(val_ds.data))
    print(len(test_ds.data))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    weights = models.ResNet18_Weights.IMAGENET1K_V1 if transfer_learning else None
    resnet18_model = models.resnet18(weights=weights, num_classes=len(classes))
    num_ftrs = resnet18_model.fc.in_features
    resnet18_model.fc = nn.Linear(num_ftrs, len(classes))

    optimizer = optim.Adam(resnet18_model.parameters(), learning_rate)
    metric = Accuracy(task="multiclass", num_classes=len(classes), average="macro")
    loss_fn = nn.CrossEntropyLoss()

    train(
        resnet18_model, 
        train_loader, val_loader, 
        loss_fn, 
        optimizer, 
        metric, 'macro_accuracy', 
        epochs
    )