import copy
import json
import sys
from torch import nn, optim
import torch
from torch.utils.data import DataLoader

from centralized.optimization import load_weights
from federated_learning.client import ClientConfig
from federated_learning.fedavg import get_fedavg
from federated_learning.fedbuff import get_fedbuff
from federated_learning.simulation import DatasetConfig, run_fl_simulation
from shared.data import OCTDLClass, OCTDLDataset, get_balancing_weights, load_octdl_data, get_transforms, prepare_dataset, get_octdl_datasets_stratified
from shared.metrics import BalancedAccuracy, F1ScoreMacro
from shared.model import get_efficientnet, get_mobilenet, get_model_by_type
from shared.training import EarlyStopping, evaluate, set_device, train, LossFnType

def try_centralized():
    classes = [OCTDLClass.AMD, OCTDLClass.NO]
    all_classes = [OCTDLClass.AMD, OCTDLClass.DME, OCTDLClass.ERM,
               OCTDLClass.NO, OCTDLClass.RAO, OCTDLClass.RVO, OCTDLClass.VID]
    
    balancing_weight = get_balancing_weights(
        classes
    )
    metrics = [BalancedAccuracy(),  F1ScoreMacro()]
    loss_fn: LossFnType = nn.CrossEntropyLoss()

    image_size = 224
    epochs = 100

    batch_size = 64
    learning_rate = 0.0005
    apply_augmentation = True
    dropout = 0.0

    train_loader, val_loader, test_loader = get_octdl_datasets_stratified(
        classes=all_classes,
        batch_size=batch_size,
    )

    print(len(train_loader.dataset))
    print(len(val_loader.dataset))
    print(len(test_loader.dataset))

    model = get_model_by_type(
        "ResNet50", True, all_classes, dropout)

    adam = optim.Adam(model.parameters(), learning_rate)

    device = set_device()

    train_gen = train(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        optimizer=adam,
        loss_fn=loss_fn,
        metrics=metrics,
        device=device,
        early_stopping=None
    )

    highest_val_f1 = float('-inf')
    final_weights = None
    for epoch_result in train_gen:
        if epoch_result is not None:
            val_f1 = epoch_result.val_metrics[F1ScoreMacro.name()]
            if val_f1 > highest_val_f1:
                print("New best F1 score:", val_f1)
                highest_val_f1 = val_f1
                final_weights = copy.deepcopy(epoch_result.model_weights)

    model.load_state_dict(final_weights)

    metrics, loss, cm = evaluate(model, test_loader, loss_fn, metrics, device=device)

    print(metrics)
    print(cm.tolist())


def try_federated():
    metrics = [BalancedAccuracy, F1ScoreMacro]
    def callback(round, loss, m):
        print(f"I AM THE CALLBACK FROM ROUND {round}, the loss is {loss}")
        print(m)

    all_classes = [OCTDLClass.AMD, OCTDLClass.NO]
    model = get_model_by_type(
            'MobileNetV2', True, all_classes, 0.2)

    device = set_device()

    n_clients = 3

    
    """
    fedavg = get_fedavg(
        n_clients,
        10,
        metrics,
        model=model, 
        checkpoint_path=None,
        optimization_mode='maximize_f1_macro',
        on_aggregate_evaluated=callback 
    )

    """
    fedbuff = get_fedbuff(
        buffer_size=3, 
        n_clients=n_clients, 
        server_lr=0.06, 
        metrics=metrics, 
        optimization_mode='maximize_f1_macro',
        model=model, 
        checkpoint_path=None,
        on_aggregate_evaluated=callback
    )

    h = run_fl_simulation(
        n_clients=n_clients,
        n_rounds=10,
        dataset_config=DatasetConfig(
            augmentation=False,
            batch_size=16,
            classes=all_classes,
            pin_memory=True
        ),
        client_config=ClientConfig(
            device=device,
            dropout=0.2,
            epochs=5,
            loss_fn_type='CrossEntropy',
            lr=0.002,
            model_type='MobileNetV2',
            transfer_learning=True,
            metrics=metrics,
            validation_batch_size=32,
            optimized=True,
        ),
        strategy=fedbuff,
        strategy_name='FedBuff'
    )

    _, val_loader, test_loader = prepare_dataset(all_classes,False,1,32)

    metrics, loss, cm = evaluate(model, test_loader, nn.CrossEntropyLoss(), [BalancedAccuracy(), F1ScoreMacro()], device)
    
    print(metrics)
    sys.exit(0)
    

if __name__ == "__main__":
    try_centralized()
   # try_federated()