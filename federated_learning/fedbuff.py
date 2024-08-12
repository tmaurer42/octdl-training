from functools import reduce
from typing import Callable, Optional, Union
from logging import ERROR, INFO

import math
from scipy.stats import norm
import numpy as np
from flwr.common import FitIns, FitRes, Parameters, NDArrays, parameters_to_ndarrays, ndarrays_to_parameters, Metrics, log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
import torch

from federated_learning.server import get_avg_metrics_fn
from federated_learning.strategy import wrap_strategy
from shared.metrics import CategoricalMetric


def random_halfnormal_variate(max):
    """
    Generate a random number between 0 and max (inclusive) using half normal distribution
    """
    if max == 0:
        return 0

    value = abs(norm.rvs(scale=max/4))

    return int(round(value % max))


class FedBuff(FedAvg):
    def init_fedbuff(self, k: int = 1, server_lr=1.0):
        self.buffer_size = k
        self.server_lr = server_lr
        self.all_parameters: dict[int, Parameters] = {}
        self.clients_param_version: dict[str, int] = {}

    def get_random_parameters(self, server_round: int, last_used_param_version: Optional[int]):
        if last_used_param_version is None:
            max_staleness = len(self.all_parameters) - 1
        else:
            max_staleness = server_round - last_used_param_version

        if max_staleness == 0:
            staleness = 0
        else:
            # Use halfnormal staleness distribution, as observed in FedBuff paper
            staleness = random_halfnormal_variate(max_staleness)
            # staleness = np.random.randint(0, max_staleness)
        param_version = server_round - staleness
        params = self.all_parameters[param_version]

        return params, staleness, param_version

    def aggregate_updates(
        self,
        results: list[tuple[ClientProxy, FitRes]]
    ):
        """
        FedBuff algorithm line 8 and 11:
        Updates are summed, divided by the buffer size and multiplied by the servers lr.
        """
        assert len(results) == self.buffer_size
        results_parameters = []

        for res in results:
            fit_res = res[1]
            params = parameters_to_ndarrays(fit_res.parameters)
            staleness_scale = (1/(1+math.sqrt(fit_res.metrics['staleness'])))
            params = [layer * staleness_scale for layer in params]
            results_parameters.append(params)

        weights_prime: NDArrays = [
            reduce(np.add, layer_updates) * self.server_lr / self.buffer_size
            for layer_updates in zip(*results_parameters)
        ]
        epsilon = 1e-5
        weights_prime = [np.where(np.abs(arr) < epsilon, 0, arr)
                         for arr in weights_prime]

        return ndarrays_to_parameters(weights_prime)

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager
    ):
        """
        Sample <self.buffer_size> clients from all participating ones.
        Each client gets randomly stale parameters.
        """
        clients_available = client_manager.num_available()
        if clients_available < self.buffer_size:
            raise ValueError(
                f"At least {self.buffer_size} clients need to be available")

        sample_size = self.buffer_size

        clients = client_manager.sample(
            num_clients=sample_size
        )

        self.all_parameters[server_round] = parameters

        client_instructions = []
        for client in clients:
            last_used_param_version = self.clients_param_version.get(
                client.cid, None)
            params, staleness, param_version = self.get_random_parameters(
                server_round, last_used_param_version)
            log(INFO, f"Client {client.cid}, last used params of round {last_used_param_version}, now uses {param_version}, staleness {staleness}")
            self.clients_param_version[client.cid] = param_version

            fit_ins = FitIns(params, {})
            fit_ins.config['staleness'] = staleness
            client_instructions.append((client, fit_ins))

        return client_instructions

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ):
        if not results:
            log(ERROR, f"No results received from clients")
            return None, {}
        if not self.accept_failures and failures:
            log(ERROR, f"One or more clients failed: {failures}")
            return None, {}

        new_parameters = self.aggregate_updates(results)

        return new_parameters, {}


def get_fedbuff(
    buffer_size: int,
    n_clients: int,
    server_lr: float,
    metrics: list[type[CategoricalMetric]],
    model: torch.nn.Module,
    checkpoint_path: str,
    on_aggregate_evaluated: Callable[[int, float, Metrics], None]
):
    evaluate_metrics_aggregation_fn = get_avg_metrics_fn(metrics)

    FedBuffWrapped = wrap_strategy(FedBuff)
    fedbuff = FedBuffWrapped(
        min_fit_clients=n_clients,
        min_evaluate_clients=n_clients,
        min_available_clients=n_clients,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn
    )

    fedbuff.init_fedbuff(k=buffer_size, server_lr=server_lr)
    fedbuff.set_custom_props(
        model, checkpoint_path, on_aggregate_evaluated)

    return fedbuff
