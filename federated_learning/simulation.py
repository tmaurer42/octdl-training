from dataclasses import dataclass

import flwr as fl
import torch

from federated_learning.strategy import FLStrategy

from .client import ClientConfig, generate_client_fn, generate_fedbuff_client_fn
from shared.data import OCTDLClass, prepare_dataset_partitioned


@dataclass
class DatasetConfig():
    classes: list[OCTDLClass]
    augmentation: bool
    batch_size: int
    n_workers: int = 0
    pin_memory: bool = False


def run_fl_simulation(
    n_clients: int,
    n_rounds: int,
    dataset_config: DatasetConfig,
    client_config: ClientConfig,
    strategy: fl.server.strategy.Strategy,
    strategy_name: FLStrategy,
) -> fl.server.History :
    train_loaders, val_loaders, _ = prepare_dataset_partitioned(
        classes=dataset_config.classes,
        augmentation=dataset_config.augmentation,
        batch_size=dataset_config.batch_size,
        n_partitions=n_clients,
        validation_batch_size=client_config.validation_batch_size,
        n_workers=dataset_config.n_workers,
        pin_memory=dataset_config.pin_memory
    )

    if strategy_name == 'FedAvg':
        client_fn = generate_client_fn(
            train_loaders=train_loaders, 
            val_loaders=val_loaders,
            classes=dataset_config.classes,
            config=client_config
        )
    if strategy_name == 'FedBuff':
        client_fn = generate_fedbuff_client_fn(
            train_loaders=train_loaders, 
            val_loaders=val_loaders,
            classes=dataset_config.classes,
            config=client_config
        )

    num_gpus = 1.0 if torch.cuda.is_available() else 0.0

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=n_clients,
        config=fl.server.ServerConfig(
            num_rounds=n_rounds
        ),
        client_resources={"num_cpus": 1, "num_gpus": num_gpus},
        strategy=strategy
    )

    return history
    
