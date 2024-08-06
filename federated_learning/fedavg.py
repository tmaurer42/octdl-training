from typing import Callable, Optional, Union

from flwr.common import Scalar, Metrics
from flwr.server.server import FitRes, Parameters
from flwr.server.strategy import FedAvg
from flwr.server.client_manager import ClientProxy
import torch

from federated_learning.server import get_avg_metrics_fn, save_parameters
from shared.metrics import CategoricalMetric


class FedAvgCustom(FedAvg):
    def set_custom_props(
        self,
        model: torch.nn.Module,
        checkpoint_path: str,
        on_aggregate_evaluated: Callable[[int, float, Metrics], None]
    ):
        self.model = model
        self.checkpoint_path = checkpoint_path
        self.on_aggregate_evaluated = on_aggregate_evaluated

    def aggregate_evaluate(
        self, 
        server_round: int, 
        results, 
        failures
    ):
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)

        if self.on_aggregate_evaluated is not None:
            self.on_aggregate_evaluated(server_round, loss, metrics)

        return loss, metrics

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate model weights using weighted average and store checkpoint"""

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super(
        ).aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None and self.model is not None:
            save_parameters(aggregated_parameters, self.model, server_round, self.checkpoint_path)

        return aggregated_parameters, aggregated_metrics


def get_fedavg(
    n_clients: int,
    metrics: list[type[CategoricalMetric]],
    model: torch.nn.Module,
    checkpoint_path: str,
    on_aggregate_evaluated: Callable[[int, float, Metrics], None]
):
    evaluate_metrics_aggregation_fn = get_avg_metrics_fn(metrics)

    federated_average = FedAvgCustom(
        min_fit_clients=n_clients,
        min_evaluate_clients=n_clients,
        min_available_clients=n_clients,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn
    )

    federated_average.set_custom_props(
        model, checkpoint_path, on_aggregate_evaluated)

    return federated_average
