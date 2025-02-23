import copy
import os
import argparse
from typing import get_args

import optuna
import torch
from torch import nn, optim

from shared.data import OCTDLClass, get_balancing_weights, prepare_dataset
from shared.metrics import BalancedAccuracy, F1ScoreMacro
from shared.training import EarlyStopping, set_device, train, TrainEpochResult, OptimizationMode, LossFnType
from shared.model import ModelType, get_model_by_type

import faulthandler

from shared.utils import delete_except, get_study_name
faulthandler.enable()


centralized_chkpts_path = os.path.join(
    '.', 'results_centralized', 'checkpoints')


def load_weights(model: nn.Module, study_name, trial_number):
    path = os.path.join(centralized_chkpts_path, study_name)
    weights_path = os.path.join(path, f"{trial_number}.pth")

    if not os.path.exists(weights_path):
        print(f"Unable to load weights from path: {weights_path}")
        return
    model.load_state_dict(torch.load(
        weights_path, map_location=torch.device('cpu'), weights_only=True))


def save_weights(study_name: str, weights: dict[str, any], trial_number):
    path = os.path.join(centralized_chkpts_path, study_name)
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(weights, os.path.join(
        path, f"{trial_number}.pth"))


def run_study(
    study_name: str,
    model_type: ModelType,
    transfer_learning: bool,
    classes: list[OCTDLClass],
    loss_fn_type: LossFnType,
    optimization_mode: OptimizationMode,
    n_jobs: int,
    n_trials: int = 100,
):
    device = set_device()

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
        train_loader, val_loader, _ = prepare_dataset(
            classes=classes,
            augmentation=apply_augmentation,
            batch_size=batch_size,
            img_target_size=image_size,
            validation_batch_size=32
        )

        # Initialize model
        model = get_model_by_type(
            model_type, transfer_learning, classes, dropout)

        adam = optim.Adam(model.parameters(), learning_rate)

        balancing_weights = get_balancing_weights(classes)
        loss_fn = None
        if loss_fn_type == 'CrossEntropy':
            loss_fn = nn.CrossEntropyLoss()
        elif loss_fn_type == 'WeightedCrossEntropy':
            loss_fn = nn.CrossEntropyLoss(
                weight=balancing_weights, label_smoothing=0.1)

        metrics = [BalancedAccuracy(), F1ScoreMacro()]

        train_gen = train(
            model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            optimizer=adam,
            loss_fn=loss_fn,
            metrics=metrics,
            early_stopping=EarlyStopping(
                from_epoch=20,
                patience=5
            ),
            print_batch_info=False,
            device=device
        )

        current_epoch = 1
        best_value: float = None
        best_epoch_result: TrainEpochResult = None

        for epoch_result in train_gen:
            if optimization_mode == 'minimize_loss':
                epoch_loss = epoch_result.val_loss
                report_value = epoch_loss

                if best_value is None or epoch_loss < best_value:
                    best_value = epoch_loss
                    best_epoch_result = copy.deepcopy(epoch_result)

            elif optimization_mode == 'maximize_f1_macro':
                epoch_f1_score = epoch_result.val_metrics[F1ScoreMacro.name()]
                report_value = epoch_f1_score

                if best_value is None or epoch_f1_score > best_value:
                    best_value = epoch_f1_score
                    best_epoch_result = copy.deepcopy(epoch_result)

            trial.report(report_value, step=current_epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            current_epoch += 1

        trial.set_user_attr('confusion_matrix',
                            best_epoch_result.val_confusion_matrix.tolist())
        trial.set_user_attr('metrics', best_epoch_result.val_metrics)
        trial.set_user_attr('val_loss', best_epoch_result.val_loss)
        save_weights(study_name, best_epoch_result.model_weights, trial.number)

        return best_value

    direction = None
    if optimization_mode == 'minimize_loss':
        direction = optuna.study.StudyDirection.MINIMIZE
    elif optimization_mode == 'maximize_f1_macro':
        direction = optuna.study.StudyDirection.MAXIMIZE

    db_path = os.path.join('results_centralized',
                           f"results.sqlite3")

    if not os.path.exists('results_centralized'):
        os.makedirs('results_centralized')

    study = optuna.create_study(
        direction=direction,
        study_name=study_name,
        storage=f"sqlite:///{db_path}",
    )

    study.optimize(objective, n_trials, n_jobs=n_jobs)

    delete_except(
        os.path.join(centralized_chkpts_path, study_name),
        f"{study.best_trial.number}.pth"
    )

    return study


def main(
    model_type: ModelType,
    class_list: list[OCTDLClass],
    transfer_learning: bool,
    loss_fn_type: LossFnType,
    optimization_mode: OptimizationMode,
    n_jobs=1
):
    study_name = get_study_name(
        class_list, model_type, transfer_learning, loss_fn_type, optimization_mode)
    run_study(
        study_name=study_name,
        classes=class_list,
        model_type=model_type,
        transfer_learning=transfer_learning,
        loss_fn_type=loss_fn_type,
        optimization_mode=optimization_mode,
        n_trials=100,
        n_jobs=n_jobs
    )
