import json
import sys
from torch import nn, optim
import torch
from torch.utils.data import DataLoader

from federated_learning.client import ClientConfig
from federated_learning.fedavg import get_fedavg
from federated_learning.fedbuff import get_fedbuff
from federated_learning.simulation import DatasetConfig, run_fl_simulation
from shared.data import OCTDLClass, OCTDLDataset, get_balancing_weights, load_octdl_data, get_transforms, prepare_dataset
from shared.metrics import BalancedAccuracy, F1ScoreMacro
from shared.model import get_efficientnet, get_mobilenet, get_model_by_type
from shared.training import EarlyStopping, evaluate, set_device, train, LossFnType

def try_centralized():
    classes = [OCTDLClass.AMD, OCTDLClass.NO]
    
    balancing_weight = get_balancing_weights(
        classes
    )
    metrics = [BalancedAccuracy(),  F1ScoreMacro()]
    loss_fn: LossFnType = nn.CrossEntropyLoss()

    image_size = 224
    epochs = 2

    batch_size = 8
    learning_rate = 0.0002557595730833588
    apply_augmentation = False
    dropout = 0.2

    train_loader, val_loader, _ = prepare_dataset(
        classes=classes,
        augmentation=apply_augmentation,
        batch_size=batch_size,
        img_target_size=image_size,
        validation_batch_size=32
    )

    print(len(train_loader.dataset))
    print(len(val_loader.dataset))
    print(len(_.dataset))

    model = get_model_by_type(
        "MobileNetV2", True, classes, dropout)

    adam = optim.Adam(model.parameters(), learning_rate)

    device = set_device()

    train_gen = train(
        model,
        train_loader=train_loader,
        #val_loader=val_loader,
        epochs=epochs,
        optimizer=adam,
        loss_fn=loss_fn,
        metrics=metrics,
        device=device
        #early_stopping=EarlyStopping(
        #    patience=5,
        #    from_epoch=20
        #)
    )

    for epoch_result in train_gen:
        if epoch_result is not None:
            print(epoch_result.val_loss)
            print(epoch_result.val_metrics)

    metrics, loss, cm = evaluate(model, val_loader, loss_fn, metrics, device=set_device())

    print(metrics)


def try_federated():
    metrics = [BalancedAccuracy, F1ScoreMacro]
    def callback(round, loss, m):
        print(f"I AM THE CALLBACK FROM ROUND {round}, the loss is {loss}")
        print(m)

    model = get_model_by_type(
            'MobileNetV2', True, [OCTDLClass.AMD, OCTDLClass.NO], 0.2)

    device = torch.device("mps")

    n_clients = 20

    """
    fedavg = get_fedavg(
        n_clients,
        10,
        metrics,
        model=model, 
        checkpoint_path='tmp_results',
        on_aggregate_evaluated=callback 
    )
    """

    fedbuff = get_fedbuff(
        buffer_size=5, 
        n_clients=n_clients, 
        server_lr=0.03, 
        metrics=metrics, 
        optimization_mode='maximize_f1_macro',
        model=model, 
        checkpoint_path=None,
        on_aggregate_evaluated=callback
    )

    h = run_fl_simulation(
        n_clients=n_clients,
        n_rounds=20,
        dataset_config=DatasetConfig(
            augmentation=False,
            batch_size=64,
            classes=[OCTDLClass.AMD, OCTDLClass.NO],
            n_workers=4
        ),
        client_config=ClientConfig(
            device=device,
            dropout=0.2,
            epochs=5,
            loss_fn_type='WeightedCrossEntropy',
            lr=0.012,
            model_type='MobileNetV2',
            transfer_learning=True,
            metrics=metrics,
            validation_batch_size=128,
            cudann_optimized=True,
        ),
        strategy=fedbuff,
        strategy_name='FedBuff'
    )

    _, val_loader, test_loader = prepare_dataset([OCTDLClass.AMD, OCTDLClass.NO],False,1,32)

    metrics, loss, cm = evaluate(model, test_loader, nn.CrossEntropyLoss(), [BalancedAccuracy(), F1ScoreMacro()], device)
    
    print(metrics)
    sys.exit(0)
    

if __name__ == "__main__":
    #try_centralized()
    try_federated()