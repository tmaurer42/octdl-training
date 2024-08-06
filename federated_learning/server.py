import os
from typing import Literal, OrderedDict

import torch
import numpy as np
import flwr as fl
from flwr.server.server import Parameters

from shared.metrics import CategoricalMetric


FLStrategy = Literal['FedAvg', 'FedBuff']


def get_avg_metrics_fn(metric_types: list[type[CategoricalMetric]]) -> fl.common.MetricsAggregationFn:
    def average_metrics(metrics):
        metrics_dict = {}
        for m in metric_types:
            metric_name = m.name()
            avg_val = np.mean([metric[metric_name] for _, metric in metrics])
            metrics_dict[metric_name] = avg_val

        return metrics_dict

    return average_metrics


def save_parameters(params: Parameters, model: torch.nn.Module, round: int, path: str):
    print(f"Saving round {round} aggregated_parameters...")

    # Convert `Parameters` to `List[np.ndarray]`
    aggregated_ndarrays: list[np.ndarray] = fl.common.parameters_to_ndarrays(
        params)

    # Convert `List[np.ndarray]` to PyTorch`state_dict`
    params_dict = zip(model.state_dict().keys(), aggregated_ndarrays)
    state_dict = OrderedDict({k: torch.tensor(v)
                                for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

    # Save the model
    checkpoint_path = os.path.join(path, f"model_round_{round}.pth")
    torch.save(model.state_dict(), checkpoint_path)