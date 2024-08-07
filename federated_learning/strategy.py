from typing import Callable, Literal, Optional, Union

from flwr.common import Scalar, Metrics
from flwr.server.server import FitRes, Parameters
from flwr.server.strategy import Strategy
from flwr.server.client_manager import ClientProxy
import torch

from federated_learning.server import save_parameters


FLStrategy = Literal['FedAvg', 'FedBuff']


def wrap_strategy(base: type[Strategy]):
    """
    Wraps a class around a Strategy, which saves the model after each round
    and adds a callback after evaluation to get the loss and the metrics.
    """
    class StrategyWrapper(base):
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
            aggregated_parameters, aggregated_metrics = super(
            ).aggregate_fit(server_round, results, failures)

            if aggregated_parameters is not None and self.model is not None:
                save_parameters(aggregated_parameters, self.model,
                                server_round, self.checkpoint_path)

            return aggregated_parameters, aggregated_metrics

    return StrategyWrapper
