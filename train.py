from typing import List, Optional

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet

from sklearn.metrics import confusion_matrix

from data import OCTDLClass, load_octdl_dataset
from metrics import BalancedAccuracy, CategoricalMetric


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
        metric_names: List[str],
        metric_values: List[float], 
        loss: float, 
        val_metric_values: Optional[List[float]] = None, 
        val_loss: Optional[float] = None,
        replace_ln: bool = False
    ):
    assert len(metric_names) == len(metric_values), "Metric names and values must have the same length."
    if val_metric_values is not None:
        assert len(metric_names) == len(val_metric_values), "Metric names and validation metric values must have the same length."
    
    train_stats = ", ".join([f"{name}: {value:0.4f}" for name, value in zip(metric_names, metric_values)]) + ", "
    train_stats += f"loss: {loss:0.4f}"
    message = train_stats

    if val_metric_values is not None and val_loss is not None:
        val_stats = ", ".join([f"val_{name}: {value:0.4f}" for name, value in zip(metric_names, val_metric_values)]) + ", "
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
        metrics: List[CategoricalMetric], 
        device: torch.device
    ):
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
        metrics: List[CategoricalMetric] = [], 
        metric_names: List[str] = [],
        patience: int = 5,
        from_epoch: int = 10
    ):
    device = set_device()
    model.to(device)
    loss_fn.to(device)

    best_val_loss = float('inf')
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
            print_stats(metric_names, train_metrics, running_loss, None, None, replace_ln=True)
        
        computed_metrics = [metric.compute().item() for metric in metrics]
        for metric in metrics:
            metric.reset()

        val_metrics, val_loss, val_confusion_matrix = evaluate(model, val_loader, loss_fn, metrics, device)

        yield val_loss

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = model.state_dict()
            best_model_confusion_matrix = val_confusion_matrix
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

    return best_val_loss, best_model_confusion_matrix



def get_resnet(
        num_classes: int = None, 
        transfer_learning: bool = False, 
        add_dense_layer: bool = False,
        dense_layer_size: int = 256,
        add_dropout_layer = False,
        dropout = 0.5
    ) -> ResNet:
    #weights = models.ResNet18_Weights.IMAGENET1K_V1 if transfer_learning else None
    resnet18_model = models.resnet18(pretrained=transfer_learning)

    if transfer_learning: 
        for params in resnet18_model.parameters():
            params.requires_grad = False

    num_ftrs = resnet18_model.fc.in_features
    additional_layers = nn.Sequential()

    if add_dense_layer:
        additional_layers.append(
            nn.Sequential(
                nn.Linear(num_ftrs, dense_layer_size),
                nn.ReLU()
            )
        )
        num_ftrs = dense_layer_size

    if add_dropout_layer:
        additional_layers.append(nn.Dropout(dropout))

    additional_layers.append(nn.Linear(num_ftrs, num_classes))

    resnet18_model.fc = additional_layers
    return resnet18_model
    

def get_transforms(img_target_size: int):
    mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

    base_transform = transforms.Compose([
        transforms.Resize((img_target_size, img_target_size)),
        transforms.Normalize(mean=mean, std=std)
    ])

    augment_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_target_size, scale=(0.8, 1.0)),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.GaussianBlur(kernel_size=(3, 5)),
        transforms.Normalize(mean=mean, std=std),
    ])

    return base_transform, augment_transform

if __name__ == "__main__":
    classes = [OCTDLClass.AMD, OCTDLClass.NO]
    transfer_learning = False
    image_size = 224

    batch_size = 32
    learning_rate = 0.0005
    epochs = 20

    mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

    val_test_transform, train_transform = get_transforms(image_size)

    train_ds, val_ds, test_ds, balancing_weights = load_octdl_dataset(
        classes,
        train_transform,
        val_test_transform
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    
    model = get_resnet(
        transfer_learning=False,
        num_classes=len(classes),
        add_dropout_layer=True,
        dropout=0.2
    )

    adam = optim.Adam(model.parameters(), learning_rate)
    
    balanced_accuracy = BalancedAccuracy()
    #cohen_kappa = CohenKappa(task="multiclass", num_classes=len(classes))
    #f1_score = F1Score(task="multiclass", num_classes=len(classes), average="macro")
    #roc_auc = AUROC(task="multiclass", num_classes=len(classes))

    cross_entropy_loss = nn.CrossEntropyLoss()
    weighted_cross_entropy_loss = nn.CrossEntropyLoss(weight=balancing_weights, label_smoothing=0.1)


    train_gen = train(
        model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        epochs=epochs,
        optimizer=adam, 
        loss_fn=weighted_cross_entropy_loss, 
        metrics=[balanced_accuracy], 
        metric_names=['balanced_accuracy'], 
    )

    while True:
        try:
            running_loss = next(train_gen)
        except StopIteration as res:
            best_val_loss, best_confusion_matrix = res.value
            print(f"Best val loss: {best_val_loss}")
            break
        