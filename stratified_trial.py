import copy
from torch import nn, optim
from shared.data import OCTDLClass, get_balancing_weights, get_octdl_datasets_stratified
from shared.metrics import BalancedAccuracy, F1ScoreMacro
from shared.model import get_model_by_type
from shared.training import evaluate, set_device, train, LossFnType

def stratified_trial():
    all_classes = [OCTDLClass.AMD, OCTDLClass.DME, OCTDLClass.ERM,
               OCTDLClass.NO, OCTDLClass.RAO, OCTDLClass.RVO, OCTDLClass.VID]
    
    metrics = [BalancedAccuracy(),  F1ScoreMacro()]
    weights = get_balancing_weights(all_classes)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    epochs = 100

    batch_size = 64
    learning_rate = 0.0005
    dropout = 0.0

    train_loader, val_loader, test_loader = get_octdl_datasets_stratified(
        classes=all_classes,
        batch_size=batch_size,
    )

    print(len(train_loader.dataset))
    print(len(val_loader.dataset))
    print(len(test_loader.dataset))

    model = get_model_by_type(
        "ResNet50", False, all_classes, dropout)

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

    metrics, _, cm = evaluate(model, test_loader, loss_fn, metrics, device=device)

    print(metrics)
    print(cm.tolist())

if __name__ == "__main__":
    stratified_trial()