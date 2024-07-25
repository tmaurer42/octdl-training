from typing import Literal
from torch import nn
from torchvision import models

from shared.data import OCTDLClass


ModelType = Literal["ResNet18", "MobileNetV2", "EfficientNetV2"]


def get_model_by_type(
    model_type: ModelType,
    transfer_learning: bool,
    classes: list[OCTDLClass],
    dropout: float
):
    if model_type == "ResNet18":
        return get_resnet(
            transfer_learning=transfer_learning,
            num_classes=len(classes),
            dropout=dropout
        )
    if model_type == "MobileNetV2":
        return get_mobilenet(
            transfer_learning=transfer_learning,
            num_classes=len(classes),
            dropout=dropout
        )

    if model_type == "EfficientNetV2":
        return get_efficientnet(
            transfer_learning=transfer_learning,
            num_classes=len(classes),
            dropout=dropout
        )


def get_resnet(
    num_classes: int,
    transfer_learning: bool = False,
    dropout=0.2
) -> models.ResNet:
    """
    Load and initialize a pytorch ResNet18 model.

    Parameters:
        num_classes (int):
            Number of classes the model should output in the last layer.
        transfer_learning (bool):
            If True, initialize the model for transfer learning. 
            First, freeze all base layers and then 
            add two fully connected layers with dropout before 
            the classification layer.
            Default is False
        dropout (float):
            Value for the dropout layers. 
            This affects the ones added when transfer_learning is True.
            If transfer_learning is False, adds a Dropout layer
            before the classification layer. 
            Default is 0.2.

    Returns:
        The configured ResNet18 model.
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if transfer_learning else None
    resnet18_model = models.resnet18(weights=weights)

    last_layer_input_size = resnet18_model.fc.in_features
    layers = []

    if transfer_learning:
        for params in resnet18_model.parameters():
            params.requires_grad = False

        dense_layer_size = last_layer_input_size // 2
        dense_layer_2_size = last_layer_input_size // 4
        layers.extend([
            nn.Linear(last_layer_input_size, dense_layer_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_layer_size, dense_layer_2_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        ])
        last_layer_input_size = dense_layer_2_size

    if not transfer_learning:
        layers.append(nn.Dropout(dropout))

    layers.append(nn.Linear(last_layer_input_size, num_classes))

    resnet18_model.fc = nn.Sequential(*layers)
    return resnet18_model


def get_mobilenet(
    num_classes: int,
    transfer_learning: bool = False,
    dropout=0.2
) -> models.MobileNetV2:
    """
    Load and initialize a pytorch MobileNetV2 model.

    Parameters:
        num_classes (int):
            Number of classes the model should output in the last layer.
        transfer_learning (bool):
            If True, initialize the model for transfer learning. 
            First, freeze all base layers and then 
            add two fully connected layers with dropout before 
            the classification layer. 
            Default is False.
        dropout (float): 
            Value for the dropout layers. 
            This affects the default dropout layer and the ones added when transfer_learning is True.
            Default is 0.2.

    Returns:
        The configured MobileNetV2 model.
    """
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V2 if transfer_learning else None
    mobilenetv2_model = models.mobilenet_v2(weights=weights)

    last_layer_input_size = mobilenetv2_model.last_channel
    layers = []

    if transfer_learning:
        for params in mobilenetv2_model.parameters():
            params.requires_grad = False

        dense_layer_size = last_layer_input_size // 2
        dense_layer_2_size = last_layer_input_size // 4
        layers.extend([
            nn.Linear(last_layer_input_size, dense_layer_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_layer_size, dense_layer_2_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        ])

        last_layer_input_size = dense_layer_2_size

    layers.append(nn.Linear(last_layer_input_size, num_classes))

    mobilenetv2_model.classifier[0] = nn.Dropout(dropout)
    mobilenetv2_model.classifier[1] = nn.Sequential(*layers)
    return mobilenetv2_model


def get_efficientnet(
    num_classes: int,
    transfer_learning: bool = False,
    dropout=0.2
) -> models.EfficientNet:
    """
    Load and initialize a pytorch EfficientNet_V2_S model.

    Parameters:
        num_classes (int):
            Number of classes the model should output in the last layer.
        transfer_learning (bool):
            If True, initialize the model for transfer learning. 
            First, freeze all base layers and then 
            add two fully connected layers with dropout before 
            the classification layer. 
            Default is False.
        dropout (float): 
            Value for the dropout layers. 
            This affects the default dropout layer and the ones added when transfer_learning is True.
            Default is 0.2.

    Returns:
        The configured EfficientNet_V2_S model.
    """
    weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if transfer_learning else None
    efficientnet_model = models.efficientnet_v2_s(weights=weights)

    last_layer_input_size = efficientnet_model.classifier[1].in_features

    layers = []

    if transfer_learning:
        for params in efficientnet_model.parameters():
            params.requires_grad = False

        dense_layer_size = last_layer_input_size // 2
        dense_layer_2_size = last_layer_input_size // 4
        layers.extend([
            nn.Linear(last_layer_input_size, dense_layer_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_layer_size, dense_layer_2_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        ])

        last_layer_input_size = dense_layer_2_size

    layers.append(nn.Linear(last_layer_input_size, num_classes))

    efficientnet_model.classifier[0] = nn.Dropout(dropout)
    efficientnet_model.classifier[1] = nn.Sequential(*layers)
    return efficientnet_model
