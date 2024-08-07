from typing import Callable

from flwr.common import Metrics
from flwr.server.strategy import FedAvg
import torch

from federated_learning.server import get_avg_metrics_fn
from federated_learning.strategy import wrap_strategy
from shared.metrics import CategoricalMetric


def get_fedavg(
    n_clients: int,
    metrics: list[type[CategoricalMetric]],
    model: torch.nn.Module,
    checkpoint_path: str,
    on_aggregate_evaluated: Callable[[int, float, Metrics], None]
):
    evaluate_metrics_aggregation_fn = get_avg_metrics_fn(metrics)

    FedAvgCustom = wrap_strategy(FedAvg)
    federated_average = FedAvgCustom(
        min_fit_clients=n_clients,
        min_evaluate_clients=n_clients,
        min_available_clients=n_clients,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn
    )

    federated_average.set_custom_props(
        model, checkpoint_path, on_aggregate_evaluated)

    return federated_average
