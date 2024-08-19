from logging import INFO
import os
from typing import Callable, Literal, Optional, Union

from flwr.common import Scalar, Metrics, log
from flwr.server.server import FitRes, Parameters
from flwr.server.strategy import Strategy
from flwr.server.client_manager import ClientProxy
import torch

from federated_learning.utils import apply_parameters


FLStrategy = Literal['FedAvg', 'FedBuff']


def wrap_strategy(base: type[Strategy]):
    """
    Wraps a class around a Strategy, which saves the model after each round
    and adds a callback after evaluation to get the loss and the metrics.
    """
    class StrategyWrapper(base):
        def init_strategy(
            self,
            model: torch.nn.Module,
            checkpoint_path: str,
            on_aggregate_evaluated: Callable[[int, float, Metrics], None]
        ):
            self.model = model
            self.checkpoint_path = checkpoint_path
            self.on_aggregate_evaluated = on_aggregate_evaluated
            self.lowest_loss = float('inf')

            if self.checkpoint_path is not None:
                if not os.path.exists(checkpoint_path):
                    os.makedirs(checkpoint_path)

        def aggregate_evaluate(
            self,
            server_round: int,
            results,
            failures
        ):
            loss, metrics = super().aggregate_evaluate(server_round, results, failures)

            if loss < self.lowest_loss:
                self.lowest_loss = loss
                if self.checkpoint_path is not None and self.model is not None:
                    log(INFO, f"Saving model with lowest loss so far: {loss}")
                    path = os.path.join(self.checkpoint_path, f"best_model.pth")
                    torch.save(self.model.state_dict(), path)

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

            is_fedbuff = base.__name__ == 'FedBuff'

            if aggregated_parameters is not None and self.model is not None:
                apply_parameters(aggregated_parameters, self.model,
                                trainable_params_only=is_fedbuff)

            return aggregated_parameters, aggregated_metrics

    return StrategyWrapper
