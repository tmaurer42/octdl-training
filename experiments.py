import os
from typing import Literal
import optuna

import torch
from torch import nn, optim
from torchvision.models import ResNet
from torch.utils.data import DataLoader

from data import OCTDLClass, OCTDLDataset, load_octdl_data
from metrics import BalancedAccuracy, F1ScoreMacro, CategoricalMetric
from train import train, get_transforms
from model import get_resnet, get_mobilenet


ModelType = Literal["ResNet18", "MobileNetV2"]


def load_weights(study_name, model: nn.Module, trial_number):
    path = os.path.join('.', 'params', study_name)
    weights_path = os.path.join(path, f"{trial_number}.pth")

    if not os.path.exists(weights_path):
        print(f"Unable to load weights from path: {weights_path}")
        return
    model.load_state_dict(torch.load(weights_path))


def save_weights(study_name: str, model: ResNet, trial_number):
    path = os.path.join('.', 'params', study_name)
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), os.path.join(path, f"{trial_number}.pth"))


def run_study(
        study_name: str,
        model_type: ModelType,
        train_data: list,
        val_data: list,
        transfer_learning: bool,
        classes: list[OCTDLClass], 
        loss_fn: nn.CrossEntropyLoss,
        metrics: list[CategoricalMetric],
        metric_names: list[str],
        n_trials: int = 100
    ):

    def objective(trial: optuna.Trial):
        image_size = 224
        epochs = 100

        # Tunable Hyperparameters
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128])
        learning_rate = trial.suggest_float("learning_rate", 0.0001, 0.1, log=True)
        apply_augmentation = trial.suggest_categorical("apply_augmentation", [True, False])
        dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)

        # Initialize data
        base_transform, train_transform = get_transforms(image_size)
        train_ds = OCTDLDataset(
            train_data, 
            classes, 
            transform=train_transform if apply_augmentation else base_transform
        )
        val_ds = OCTDLDataset(val_data, classes, transform=base_transform)
    
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        # Initialize model
        if model_type == "ResNet18":
            model = get_resnet(
                transfer_learning=transfer_learning,
                num_classes=len(classes),
                dropout=dropout
            )
        if model_type == "MobileNetV2":
            model = get_mobilenet(
                transfer_learning=transfer_learning,
                num_classes=len(classes),
                dropout=dropout
            )

        adam = optim.Adam(model.parameters(), learning_rate)

        train_gen = train(
            model, 
            train_loader=train_loader, 
            val_loader=val_loader, 
            epochs=epochs,
            optimizer=adam, 
            loss_fn=loss_fn, 
            metrics=metrics, 
            metric_names=metric_names, 
            patience=5,
            from_epoch=20
        )

        current_epoch = 1
        while True:
            try:
                running_loss = next(train_gen)
                trial.report(running_loss, step=current_epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
                current_epoch += 1
            except StopIteration as res:
                best_val_loss, best_confusion_matrix, best_model_metrics = res.value

                # Set confusion_matrix and metrics in the trial to see it in the dashboard
                trial.set_user_attr('best_confusion_matrix', best_confusion_matrix.tolist())
                metrics_dict =  {f"val_{name}": value for name, value in zip(metric_names, best_model_metrics)}
                trial.set_user_attr('best_model_metrics', metrics_dict)

                # Save the weights for testing the model
                save_weights(study_name, model, trial.number)
                break

        return best_val_loss

    study = optuna.create_study(
        direction=optuna.study.StudyDirection.MINIMIZE,
        study_name=study_name,
        storage="sqlite:///results.sqlite3",
    )

    study.optimize(objective, n_trials)

    return study


def get_study_name(
        classes: list[OCTDLClass], 
        model: ModelType, 
        transfer_learning: bool,
        loss: nn.CrossEntropyLoss
    ):
    classes_str = f"({', '.join([cls.name for cls in classes])})"
    transfer_learning_str = "transfer" if transfer_learning else "no transfer"
    loss_str = "WeightedCrossEntropy" if loss.weight is not None else "CrossEntropy"

    return f"{classes_str} | {model} | {transfer_learning_str} | {loss_str}"


def main():
    metrics = [BalancedAccuracy(),  F1ScoreMacro()]
    metric_names = ['balanced_accuracy', 'f1_score']

    use_cases = [
        [OCTDLClass.AMD, OCTDLClass.NO]
    ]
    models = ["ResNet18"]
    loss_fns = [
        nn.CrossEntropyLoss(),
        nn.CrossEntropyLoss(weight=balancing_weights, label_smoothing=0.1)
    ]

    for class_list in use_cases:
        train_data, val_data, test_data, balancing_weights = load_octdl_data(
            class_list
        )
        for model_type in models:
            for transfer_learning in [False, True]:
                studies: list[optuna.Study] = []
                for loss_fn in loss_fns:
                    study_name = get_study_name(class_list, model_type, transfer_learning, loss_fn)
                    study = run_study(
                        study_name=study_name,
                        classes=class_list,
                        model_type=model_type,
                        train_data=train_data,
                        val_data=val_data,
                        transfer_learning=transfer_learning,
                        loss_fn=loss_fn,
                        metrics=metrics,
                        metric_names=metric_names,
                        n_trials=100
                    )
                    studies.append(study)
                
                for study in studies:
                    best_metrics = study.best_trial.user_attrs('metrics')


if __name__ == "__main__":
    main()
                    



