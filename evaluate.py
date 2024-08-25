from dataclasses import dataclass
import os
from typing import Optional

import optuna
import torch
from torch import nn
import torch.utils
import torch.utils.data
from sklearn import metrics

from centralized.optimization import get_study_name, load_weights
from federated_learning.strategy import FLStrategy
from federated_learning.optimization import load_weights as load_weigths_fl
from shared.data import OCTDLClass, OCTDLDataset, get_transforms, load_octdl_data, prepare_dataset
from shared.metrics import BalancedAccuracy, CategoricalMetric, F1ScoreMacro, balanced_accuracy
from shared.model import ModelType, get_model_by_type
from shared.training import set_device, LossFnType, OptimizationMode
from shared.utils import get_fl_study_name


@dataclass
class FLEvalParameters:
    strategy: FLStrategy
    n_clients: int
    n_clients_per_round: Optional[int]


def get_result_db_name(
    model_type: ModelType,
    classes: list[OCTDLClass],
    optimization_mode: OptimizationMode
):
    classes_str = '-'.join([cls.name for cls in classes])
    return f"results_{model_type}_{classes_str}_{optimization_mode}.sqlite3"


def get_study(
    classes: list[OCTDLClass],
    model_type: ModelType,
    transfer_learning: bool,
    loss_fn_type: LossFnType,
    optimization_mode: OptimizationMode,
    fl_eval_parameters: Optional[FLEvalParameters] = None
):
    if fl_eval_parameters is not None:
        study_name = get_fl_study_name(
            classes,
            model_type,
            transfer_learning=transfer_learning,
            loss_fn_type=loss_fn_type,
            optimization_mode=optimization_mode,
            n_clients=fl_eval_parameters.n_clients,
            n_clients_per_round=fl_eval_parameters.n_clients_per_round
        )
        results_path = f"results_{fl_eval_parameters.strategy}"
    else:
        study_name = get_study_name(
            classes,
            model_type,
            transfer_learning=transfer_learning,
            loss_fn_type=loss_fn_type,
            optimization_mode=optimization_mode
        )
        results_path = "results_centralized"

    db_name = "results.sqlite3"
    db_url = f"sqlite:///{os.path.join(results_path, db_name)}"
    study: optuna.Study = optuna.load_study(
        study_name=study_name, storage=db_url)

    return study


def evaluate(
    classes: list[OCTDLClass],
    model_type: ModelType,
    transfer_learning: bool,
    optimization_mode: OptimizationMode,
    loss_fn_type: LossFnType,
    fl_eval_parameters: Optional[FLEvalParameters] = None
):
    study = get_study(
        classes, model_type, transfer_learning,
        loss_fn_type, optimization_mode, fl_eval_parameters)

    study_name = study.study_name
    best_trial = study.best_trial

    _, _, test_loader = prepare_dataset(
        classes=classes, augmentation=False, batch_size=1, validation_batch_size=32)

    model = get_model_by_type(model_type, transfer_learning, classes, 0.0)

    if fl_eval_parameters is None:
        load_weights(model, study_name, best_trial.number)
    else:
        study_values = study.best_trial.intermediate_values
        best_round = min(study_values, key=study_values.get) + 1
        load_weigths_fl(fl_eval_parameters.strategy, model, study_name,
                        study.best_trial.number, best_round)

    all_preds = []
    all_labels = []
    device = set_device()
    model.eval()
    model.to(device)
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs.data, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    bal_acc = balanced_accuracy(all_preds, all_labels)
    f1score_macro = metrics.f1_score(all_labels, all_preds, average='macro')
    f1score_amd = metrics.f1_score(
        all_labels, all_preds, average='binary', pos_label=0)

    classes_str = ','.join([cls.name for cls in classes])

    print()
    print(
        f">> Evaluation on testset for [{classes_str}] using model {model_type}",
        f"with{'' if transfer_learning else 'out'} transfer learning."
    )
    if fl_eval_parameters is not None:
        print(
            f"Federated Learning Strategy: {fl_eval_parameters.strategy}.",
            f"n_clients: {fl_eval_parameters.n_clients}",
            f"clients per update: {fl_eval_parameters.n_clients_per_round}" if fl_eval_parameters.n_clients_per_round is not None else "",
            f"Lowest aggregated validation loss reached at round {best_round}"
        )
    print(f"Optimization mode: {optimization_mode}")
    print(f"Study name: {study.study_name}")
    print()
    print(f"Balanced Accuracy: {bal_acc:0.4f}")
    print(f"F1 Score averaged: {f1score_macro:0.4f}")
    print(f"F1 Score for AMD: {f1score_amd:0.4f}")
    print()
