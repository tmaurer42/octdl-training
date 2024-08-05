import json
from torch import nn, optim
from torch.utils.data import DataLoader

from shared.data import OCTDLClass, OCTDLDataset, get_balancing_weights, load_octdl_data, get_transforms, prepare_dataset
from shared.metrics import BalancedAccuracy, F1ScoreMacro
from shared.model import get_efficientnet, get_mobilenet, get_model_by_type
from shared.training import EarlyStopping, evaluate, set_device, train, LossFnType

if __name__ == "__main__":
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
        img_target_size=image_size
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
