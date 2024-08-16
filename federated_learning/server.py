from collections import OrderedDict
import os

import torch
import numpy as np
import flwr as fl
from flwr.server.server import Parameters

from shared.metrics import CategoricalMetric


def get_avg_metrics_fn(metric_types: list[type[CategoricalMetric]]) -> fl.common.MetricsAggregationFn:
    def average_metrics(metrics):
        metrics_dict = {}
        for m in metric_types:
            metric_name = m.name()
            avg_val = np.mean([metric[metric_name] for _, metric in metrics])
            metrics_dict[metric_name] = avg_val

        return metrics_dict

    return average_metrics


def apply_parameters(params: Parameters, model: torch.nn.Module, trainable_params_only=False):
    if trainable_params_only:
        new_state_dict = OrderedDict()
        trainable_param_names = [
            name for name, p in model.named_parameters() if p.requires_grad]
        params_ndarrays = fl.common.parameters_to_ndarrays(params)
        assert len(params_ndarrays) == len(trainable_param_names)

        for name, param in model.state_dict().items():
            if name in trainable_param_names:
                p = params_ndarrays.pop(0)
                new_state_dict[name] = torch.tensor(p)
            else:
                new_state_dict[name] = param
    else:
        aggregated_ndarrays: list[np.ndarray] = fl.common.parameters_to_ndarrays(
            params)

        params_dict = zip(model.state_dict().keys(), aggregated_ndarrays)
        new_state_dict = OrderedDict({k: torch.tensor(v)
                                      for k, v in params_dict})

    model.load_state_dict(new_state_dict, strict=True)
