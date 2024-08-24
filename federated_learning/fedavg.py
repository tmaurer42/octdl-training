from typing import Callable

from flwr.common import Metrics
from flwr.server.strategy import FedAvg
import torch

from federated_learning.utils import get_avg_metrics_fn
from federated_learning.strategy import wrap_strategy
from shared.metrics import CategoricalMetric
from shared.training import OptimizationMode


def get_fedavg(
    n_clients: int,
    n_fit_clients_per_round: int,
    metrics: list[type[CategoricalMetric]],
    optimization_mode: OptimizationMode,
    model: torch.nn.Module,
    checkpoint_path: str,
    on_aggregate_evaluated: Callable[[int, float, Metrics], None]
):
    evaluate_metrics_aggregation_fn = get_avg_metrics_fn(metrics)

    fraction_fit = n_fit_clients_per_round / n_clients

    FedAvgCustom = wrap_strategy(FedAvg)
    federated_average = FedAvgCustom(
        fraction_fit=fraction_fit,
        min_evaluate_clients=n_clients,
        min_available_clients=n_clients,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn
    )

    federated_average.init_strategy(
        optimization_mode, model, checkpoint_path, on_aggregate_evaluated)

    return federated_average
