from logging import INFO
import time
import os

import optuna
import torch
from flwr.common.logger import log

from federated_learning.client import ClientConfig
from federated_learning.fedavg import get_fedavg
from federated_learning.fedbuff import get_fedbuff
from federated_learning.strategy import FLStrategy
from federated_learning.simulation import DatasetConfig, run_fl_simulation
from shared.data import OCTDLClass
from shared.metrics import BalancedAccuracy, F1ScoreMacro
from shared.model import ModelType, get_model_by_type
from shared.training import LossFnType, OptimizationMode, set_device
from shared.utils import delete_except, get_fl_study_name


def get_results_path(fl_strategy: FLStrategy):
    return os.path.join('.', f'results_{fl_strategy}')


def get_checkpoints_folder_path(study_name: str, fl_strategy: FLStrategy):
    return os.path.join(
        get_results_path(fl_strategy), 'checkpoints', study_name)


def load_weights(fl_strategy: FLStrategy, model: torch.nn.Module, study_name, trial_number, round):
    path = os.path.join(
        get_checkpoints_folder_path(study_name, fl_strategy),
        f"trial_{trial_number}"
    )
    weights_path = os.path.join(path, f"model_round_{round}.pth")

    if not os.path.exists(weights_path):
        print(f"Unable to load weights from path: {weights_path}")
        return
    model.load_state_dict(torch.load(
        weights_path, map_location=torch.device('cpu')))


def run_study(
    study_name: str,
    fl_strategy: FLStrategy,
    n_clients: int,
    n_rounds: int,
    model_type: ModelType,
    transfer_learning: bool,
    classes: list[OCTDLClass],
    loss_fn_type: LossFnType,
    optimization_mode: OptimizationMode,
    buffer_size: int = None,
    n_jobs: int = 1,
    n_trials: int = 100,
):
    if optimization_mode == 'maximize_f1_macro':
        raise ValueError("maximize_f1_macro is not supported in FL")

    results_path = get_results_path(fl_strategy)

    def objective(trial: optuna.Trial):
        device = set_device()

        # Tunable Hyperparameters
        batch_size = trial.suggest_categorical(
            "batch_size", [8, 16, 32, 64, 128])
        learning_rate = trial.suggest_float(
            "learning_rate", 0.0001, 0.1, log=True)
        apply_augmentation = trial.suggest_categorical(
            "apply_augmentation", [True, False])
        dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)
        local_epochs = trial.suggest_int("local_epochs", 1, 10, step=1)

        if fl_strategy == 'FedBuff':
            server_lr = trial.suggest_float("server_lr", 0.0001, 2, log=True)

        # Initialize model
        model = get_model_by_type(
            model_type, transfer_learning, classes, dropout)

        metrics = [BalancedAccuracy, F1ScoreMacro]

        checkpoints_folder_path = get_checkpoints_folder_path(
            study_name, fl_strategy)
        checkpoints_path = os.path.join(
            checkpoints_folder_path, f"trial_{trial.number}")

        if not os.path.exists(checkpoints_path):
            os.makedirs(checkpoints_path)

        def on_server_evaluate(round, loss, metrics):
            log(INFO, f"eval loss: {loss}, eval metrics: {metrics}")
            trial.report(loss, round - 1)
            if trial.should_prune():
                raise optuna.TrialPruned

        if fl_strategy == 'FedAvg':
            strategy = get_fedavg(n_clients, metrics, model,
                                  checkpoints_path, on_server_evaluate)
        if fl_strategy == 'FedBuff':
            strategy = get_fedbuff(
                buffer_size,
                n_clients,
                server_lr,
                metrics,
                model,
                checkpoints_path,
                on_server_evaluate
            )

        try:
            # Wait a bit, so the server from the previous trial can shut down
            time.sleep(2)
            log(INFO, trial.params)
            history = run_fl_simulation(
                n_clients=n_clients,
                n_rounds=n_rounds,
                dataset_config=DatasetConfig(
                    augmentation=apply_augmentation,
                    batch_size=batch_size,
                    classes=classes
                ),
                client_config=ClientConfig(
                    device=device,
                    dropout=dropout,
                    epochs=local_epochs,
                    loss_fn_type=loss_fn_type,
                    lr=learning_rate,
                    model_type=model_type,
                    transfer_learning=transfer_learning,
                    metrics=metrics
                ),
                strategy=strategy,
                strategy_name=fl_strategy
            )
        except Exception as ex:
            # run_fl_simulation throws a custom error when the optuna error is raised
            # so we check again if the current trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned
            else:
                raise ex

        bal_accuracies = history.metrics_distributed[BalancedAccuracy.name()]
        f1_macro_scores = history.metrics_distributed[F1ScoreMacro.name()]

        losses = history.losses_distributed
        lowest_loss = min(losses, key=lambda round_loss: round_loss[1])

        lowest_loss_round = lowest_loss[0]
        bal_accuracies = history.metrics_distributed[BalancedAccuracy.name()]
        f1_macro_scores = history.metrics_distributed[F1ScoreMacro.name()]
        trial.set_user_attr(BalancedAccuracy.name(),
                            bal_accuracies[lowest_loss_round - 1])
        trial.set_user_attr(F1ScoreMacro.name(),
                            f1_macro_scores[lowest_loss_round - 1])

        return lowest_loss[1]

    db_path = os.path.join(results_path, f"results.sqlite3")
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    study = optuna.create_study(
        direction=optuna.study.StudyDirection.MINIMIZE,
        study_name=study_name,
        storage=f"sqlite:///{db_path}",
    )

    study.optimize(objective, n_trials, n_jobs=n_jobs)

    delete_except(
        get_checkpoints_folder_path(study.study_name, fl_strategy),
        f"trial_{study.best_trial.number}"
    )

    return study


def main(
    model_type: ModelType,
    class_list: list[OCTDLClass],
    transfer_learning: bool,
    loss_fn_type: LossFnType,
    optimization_mode: OptimizationMode,
    fl_strategy: FLStrategy,
    n_clients: int,
    n_rounds: int,
    buffer_size: int = None,
    n_jobs=1
):
    study_name = get_fl_study_name(
        class_list,
        model_type,
        transfer_learning,
        loss_fn_type,
        optimization_mode,
        n_clients,
        buffer_size
    )
    run_study(
        study_name=study_name,
        classes=class_list,
        model_type=model_type,
        transfer_learning=transfer_learning,
        loss_fn_type=loss_fn_type,
        fl_strategy=fl_strategy,
        buffer_size=buffer_size,
        n_clients=n_clients,
        n_rounds=n_rounds,
        n_trials=100,
        optimization_mode='minimize_loss',
        n_jobs=n_jobs,
    )
