from dataclasses import dataclass

import flwr as fl

from .client import ClientConfig, generate_client_fn
from shared.data import OCTDLClass, prepare_dataset_partitioned


@dataclass
class DatasetConfig():
    classes: list[OCTDLClass]
    augmentation: bool
    batch_size: int


def run_fl_simulation(
    n_clients: int,
    n_rounds: int,
    dataset_config: DatasetConfig,
    client_config: ClientConfig,
    strategy: fl.server.strategy.Strategy = None,
) -> fl.server.History :
    train_loaders, val_loaders, _ = prepare_dataset_partitioned(
        classes=dataset_config.classes,
        augmentation=dataset_config.augmentation,
        batch_size=dataset_config.batch_size,
        n_partitions=n_clients,
    )

    client_fn = generate_client_fn(
        train_loaders=train_loaders, 
        val_loaders=val_loaders,
        classes=dataset_config.classes,
        config=client_config
    )

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=n_clients,
        config=fl.server.ServerConfig(
            num_rounds=n_rounds
        ),
        strategy=strategy,
    )

    return history
    