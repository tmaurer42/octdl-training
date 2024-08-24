import copy
from logging import INFO
import os
from typing import Callable, Literal, Optional, Union

from flwr.common import Scalar, Metrics, log
from flwr.server.server import FitRes, Parameters
from flwr.server.strategy import Strategy
from flwr.server.client_manager import ClientProxy
import torch

from federated_learning.utils import apply_parameters
from shared.metrics import F1ScoreMacro
from shared.training import OptimizationMode


FLStrategy = Literal['FedAvg', 'FedBuff']


def wrap_strategy(base: type[Strategy]):
    """
    Wraps a class around a Strategy, which saves the model after each round
    and adds a callback after evaluation to get the loss and the metrics.
    """
    class StrategyWrapper(base):
        def init_strategy(
            self,
            optimization_mode: Optional[OptimizationMode],
            model: Optional[torch.nn.Module],
            checkpoint_path: Optional[str],
            on_aggregate_evaluated: Optional[Callable[[
                int, float, Metrics], None]]
        ):
            self.optimization_mode: OptimizationMode = optimization_mode
            self.model = model
            self.checkpoint_path = checkpoint_path
            self.on_aggregate_evaluated = on_aggregate_evaluated
            self.lowest_loss = float('inf')
            self.highest_metric = float('-inf')

            self.last_aggregated_parameters: Parameters

            if self.checkpoint_path is not None:
                if not os.path.exists(checkpoint_path):
                    os.makedirs(checkpoint_path)

        def save_model(self):
            if self.checkpoint_path is not None and self.model is not None:
                path = os.path.join(self.checkpoint_path, f"best_model.pth")
                torch.save(self.model.state_dict(), path)

        def aggregate_evaluate(
            self,
            server_round: int,
            results,
            failures
        ):
            loss, metrics = super().aggregate_evaluate(server_round, results, failures)

            is_fedbuff = base.__name__ == 'FedBuff'
            if self.optimization_mode is not None:

                if self.optimization_mode == 'minimize_loss':
                    if loss < self.lowest_loss:
                        self.lowest_loss = loss
                        apply_parameters(self.last_aggregated_parameters, self.model,
                                         trainable_params_only=is_fedbuff)
                        if self.checkpoint_path is not None and self.model is not None:
                            log(INFO,
                                f"Saving model with lowest loss so far: {loss}")
                            self.save_model()

                if self.optimization_mode == 'maximize_f1_macro':
                    f1_macro = metrics[F1ScoreMacro.name()]
                    if f1_macro > self.highest_metric:
                        self.highest_metric = f1_macro
                        apply_parameters(self.last_aggregated_parameters, self.model,
                                         trainable_params_only=is_fedbuff)
                        if self.checkpoint_path is not None and self.model is not None:
                            log(
                                INFO, f"Saving model with highest {F1ScoreMacro.name()} so far: {f1_macro}")
                            self.save_model()

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

            self.last_aggregated_parameters = copy.deepcopy(
                aggregated_parameters)

            return aggregated_parameters, aggregated_metrics

    return StrategyWrapper
