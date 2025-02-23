import copy
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

from federated_learning.utils import get_avg_metrics_fn
from federated_learning.strategy import wrap_strategy
from shared.metrics import CategoricalMetric
from shared.training import OptimizationMode


def random_halfnormal_variate(max):
    """
    Generate a random number between 0 and max (inclusive) using half normal distribution
    """
    if max == 0:
        return 0

    value = float('inf')
    while value > max + 1e-7:
        value = int(math.floor(abs(norm.rvs(scale=max/4))))

    return value


class FedBuff(FedAvg):
    def init_fedbuff(self, k: int = 1, server_lr=1.0):
        self.buffer_size = k
        self.server_lr = server_lr
        self.all_parameters: dict[int, Parameters] = {}
        self.clients_param_version: dict[str, int] = {}

    def get_random_parameters(self, server_round: int, last_used_param_version: Optional[int]):
        """
        For a client, get random paramters from all possible parameter versions
        based on the current server round and the last used parameters by the clients.
        """
        if last_used_param_version is None:
            max_staleness = len(self.all_parameters) - 1
        else:
            max_staleness = server_round - last_used_param_version - 1

        if max_staleness == 0:
            staleness = 0
        else:
            # Use halfnormal staleness distribution, as observed in FedBuff paper
            staleness = random_halfnormal_variate(max_staleness)
        param_version = server_round - staleness
        params = self.all_parameters[param_version]

        return params, staleness, param_version

    def aggregate_buffer(
        self,
        results: list[tuple[ClientProxy, FitRes]]
    ):
        """
        FedBuff algorithm line 8 and 11:
        Updates are summed and divided by the buffer size.
        """
        results_parameters = []

        for res in results:
            fit_res = res[1]
            params = parameters_to_ndarrays(fit_res.parameters)
            staleness_scale = 1/(1+math.sqrt(fit_res.metrics['staleness']))
            params = [layer * staleness_scale for layer in params]
            results_parameters.append(params)

        weights_prime: NDArrays = [
            reduce(np.add, layer_updates) / self.buffer_size
            for layer_updates in zip(*results_parameters)
        ]
        return weights_prime

    def update_global_params(self, parameters: Parameters, buffer: NDArrays):
        """
        FedBuff algorithm line 14:
        The buffer is multiplied with the server lr and then subtracted from the current model.
        """
        parameters_ndarrays = parameters_to_ndarrays(parameters)

        new_parameters_ndarrays = [weights - self.server_lr *
                                   update for weights, update in zip(parameters_ndarrays, buffer)]

        return ndarrays_to_parameters(new_parameters_ndarrays)


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

        self.all_parameters[server_round] = copy.copy(parameters)

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

        min_param_version = min(list(self.clients_param_version.values()))
        param_versions = list(self.all_parameters.keys())
        for param_version in param_versions:
            if param_version < min_param_version:
                self.all_parameters[param_version] = None

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

        buffer = self.aggregate_buffer(results)
        new_parameters = self.update_global_params(
            self.all_parameters[server_round], buffer)

        return new_parameters, {}


def get_fedbuff(
    n_clients: int,
    buffer_size: int,
    server_lr: float,
    metrics: list[type[CategoricalMetric]],
    optimization_mode: OptimizationMode,
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

    fedbuff.init_strategy(
        optimization_mode, model, checkpoint_path, on_aggregate_evaluated)
    fedbuff.init_fedbuff(k=buffer_size, server_lr=server_lr)

    return fedbuff
