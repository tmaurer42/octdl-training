import os
import optuna

import pandas as pd
import numpy as np

import torch
from torch import nn, optim
from torchvision.models import ResNet
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, F1Score

from data import OCTDLClass, load_octdl_dataset
from train import get_resnet, get_transforms, train

STUDY_NAME = "resnet18_amd-no_weighted-loss"

def save_model_and_confusion_matrix(model: ResNet, confusion_matrix: np.ndarray, trail_number):
    path = os.path.join('.', 'params', STUDY_NAME)
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), os.path.join(path, f"{trail_number}.pth"))

    if confusion_matrix is not None:
        cm_df = pd.DataFrame(confusion_matrix)
        cm_df.to_csv(os.path.join(path, f"{trail_number}_confusion_matrix.csv"), index=True)

def objective(trial: optuna.Trial):
    classes = [OCTDLClass.AMD, OCTDLClass.NO]
    transfer_learning = False
    image_size = 224
    epochs = 50

    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    learning_rate = trial.suggest_float("learning_rate", 0.0001, 0.1, log=True)
    apply_augmentation = trial.suggest_categorical("apply_augmentation", [True, False])
    dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)

    base_transform, train_transform = get_transforms(image_size)
    train_ds, val_ds, _, balancing_weights = load_octdl_dataset(
        classes,
        train_transform if apply_augmentation else base_transform,
        base_transform
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = get_resnet(
        transfer_learning=transfer_learning,
        num_classes=len(classes),
        add_dropout_layer=True,
        dropout=dropout
    )
    adam = optim.Adam(model.parameters(), learning_rate)
    weighted_cross_entropy_loss = nn.CrossEntropyLoss(weight=balancing_weights, label_smoothing=0.1)

    balanced_accuracy = Accuracy(task="multiclass", num_classes=len(classes), average="macro")
    f1_score = F1Score(task="multiclass", num_classes=len(classes), average="macro")

    train_gen = train(
        model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        epochs=epochs,
        optimizer=adam, 
        loss_fn=weighted_cross_entropy_loss, 
        metrics=[balanced_accuracy, f1_score], 
        metric_names=['balanced_accuracy', 'f1-score'], 
    )

    best_val_loss = None
    best_confusion_matrix = None
    current_epoch = 1

    while True:
        try:
            running_loss = next(train_gen)
            trial.report(running_loss, step=current_epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
            current_epoch += 1
        except StopIteration as res:
            best_val_loss, best_confusion_matrix = res.value
            break

    save_model_and_confusion_matrix(model, best_confusion_matrix, trial.number)

    return best_val_loss

study = optuna.create_study(
    direction=optuna.study.StudyDirection.MINIMIZE,
    study_name=STUDY_NAME,
    storage="sqlite:///db.sqlite3"
)
study.optimize(objective, 100)