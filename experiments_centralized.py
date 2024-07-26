import os
import argparse
from typing import Literal, get_args

import optuna
import torch
from torch import nn, optim
from torchvision.models import ResNet
from torch.utils.data import DataLoader

from shared.data import OCTDLClass, OCTDLDataset, get_balancing_weights, load_octdl_data, get_transforms
from shared.metrics import BalancedAccuracy, F1ScoreMacro, CategoricalMetric
from train_centralized import train
from shared.model import ModelType, get_model_by_type

import faulthandler
faulthandler.enable()


OptimizationMode = Literal["minimize_loss", "maximize_f1_macro"]


centralized_chkpts_path = os.path.join(
    '.', 'results_centralized', 'checkpoints')


def load_weights(model: nn.Module, study_name, trial_number):
    path = os.path.join(centralized_chkpts_path, study_name)
    weights_path = os.path.join(path, f"{trial_number}.pth")

    if not os.path.exists(weights_path):
        print(f"Unable to load weights from path: {weights_path}")
        return
    model.load_state_dict(torch.load(
        weights_path, map_location=torch.device('cpu')))


def save_weights(study_name: str, model: ResNet, trial_number):
    path = os.path.join(centralized_chkpts_path, study_name)
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), os.path.join(
        path, f"{trial_number}.pth"))


def run_study(
    study_name: str,
    model_type: ModelType,
    train_data: list,
    val_data: list,
    transfer_learning: bool,
    classes: list[OCTDLClass],
    loss_fn: nn.CrossEntropyLoss,
    metrics: list[CategoricalMetric],
    optimization_mode: OptimizationMode,
    n_trials: int = 100
):
    metric_names = [m.name for m in metrics]

    def objective(trial: optuna.Trial):
        image_size = 224
        epochs = 100

        # Tunable Hyperparameters
        batch_size = trial.suggest_categorical(
            "batch_size", [8, 16, 32, 64, 128])
        learning_rate = trial.suggest_float(
            "learning_rate", 0.0001, 0.1, log=True)
        apply_augmentation = trial.suggest_categorical(
            "apply_augmentation", [True, False])
        dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)

        # Initialize data
        base_transform, train_transform = get_transforms(image_size)
        train_ds = OCTDLDataset(
            train_data,
            classes,
            transform=train_transform if apply_augmentation else base_transform
        )
        val_ds = OCTDLDataset(val_data, classes, transform=base_transform)

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        # Initialize model
        model = get_model_by_type(
            model_type, transfer_learning, classes, dropout)

        adam = optim.Adam(model.parameters(), learning_rate)

        train_gen = train(
            model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            optimizer=adam,
            loss_fn=loss_fn,
            metrics=metrics,
            patience=5,
            from_epoch=20,
            print_batch_info=False
        )

        current_epoch = 1
        best_value: float = None

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
                trial.set_user_attr('confusion_matrix',
                                    best_confusion_matrix.tolist())
                metrics_dict = {f"val_{name}": value for name,
                                value in zip(metric_names, best_model_metrics)}
                trial.set_user_attr('metrics', metrics_dict)

                # Save the weights for testing the model
                save_weights(study_name, model, trial.number)

                if optimization_mode == 'minimize_loss':
                    best_value = best_val_loss
                elif optimization_mode == 'maximize_f1_macro':
                    f1_score_macro = F1ScoreMacro()
                    best_value = metrics_dict[f"val_{f1_score_macro.name}"]

                break

        return best_value

    classes_str = '-'.join([cls.name for cls in classes])

    direction = None
    if optimization_mode == 'minimize_loss':
        direction = optuna.study.StudyDirection.MINIMIZE
    elif optimization_mode == 'maximize_f1_macro':
        direction = optuna.study.StudyDirection.MAXIMIZE

    db_path = os.path.join('results_centralized',
                           f"results_{model_type}_{classes_str}_{optimization_mode}.sqlite3")
    study = optuna.create_study(
        direction=direction,
        study_name=study_name,
        storage=f"sqlite:///{db_path}",
    )

    study.optimize(objective, n_trials, n_jobs=4)

    return study


def get_study_name(
    classes: list[OCTDLClass],
    model: ModelType,
    transfer_learning: bool,
    loss: nn.CrossEntropyLoss,
    optimization_mode: OptimizationMode
):
    classes_str = f"{'-'.join([cls.name for cls in classes])}"
    transfer_learning_str = "transfer" if transfer_learning else "no-transfer"
    loss_str = "WeightedCrossEntropy" if loss.weight is not None else "CrossEntropy"

    return f"{classes_str}_{model}_{optimization_mode}_{transfer_learning_str}_{loss_str}"


def main(
    model_type: ModelType,
    class_list: list[OCTDLClass],
    transfer_learning: bool,
    optimization_mode: OptimizationMode
):
    metrics = [BalancedAccuracy(), F1ScoreMacro()]

    train_data, val_data, _ = load_octdl_data(class_list)
    balancing_weights = get_balancing_weights(class_list)

    loss_fns = [
        nn.CrossEntropyLoss(),
        nn.CrossEntropyLoss(weight=balancing_weights, label_smoothing=0.1)
    ]
    for loss_fn in loss_fns:
        study_name = get_study_name(
            class_list, model_type, transfer_learning, loss_fn, optimization_mode)
        run_study(
            study_name=study_name,
            classes=class_list,
            model_type=model_type,
            train_data=train_data,
            val_data=val_data,
            transfer_learning=transfer_learning,
            loss_fn=loss_fn,
            metrics=metrics,
            optimization_mode=optimization_mode,
            n_trials=100
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run model training with specified model type and class list.")
    parser.add_argument('--transfer_learning', action='store_true',
                        help='Whether to use transfer learning or not.')
    model_type_mode_choices = list(get_args(ModelType))
    parser.add_argument('--model_type', type=str, required=True, choices=model_type_mode_choices,
                        help=f"Type of model to use")
    parser.add_argument('--class_list', type=str, required=True,
                        help="Comma-separated list of classes.")
    optimization_mode_choices = list(get_args(OptimizationMode))
    parser.add_argument('--optimization_mode', required=True,
                        choices=optimization_mode_choices, help="Which optimization mode to use")

    args = parser.parse_args()

    transfer_learning = args.transfer_learning
    class_list_str = args.class_list.split(',')
    class_list = [getattr(OCTDLClass, cls) for cls in class_list_str]

    main(args.model_type, class_list, transfer_learning, args.optimization_mode)
