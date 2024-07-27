import json
from torch import nn, optim
from torch.utils.data import DataLoader

from experiments_centralized import LossFnType
from shared.data import OCTDLClass, OCTDLDataset, get_balancing_weights, load_octdl_data, get_transforms
from shared.metrics import BalancedAccuracy, F1ScoreMacro
from shared.model import get_efficientnet, get_mobilenet, get_model_by_type
from train_centralized import train

if __name__ == "__main__":
    classes = [OCTDLClass.AMD, OCTDLClass.NO]
    train_data, val_data, test_data = load_octdl_data(
        classes
    )
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

    base_transform, train_transform = get_transforms(image_size)
    train_ds = OCTDLDataset(
        train_data,
        classes,
        transform=train_transform if apply_augmentation else base_transform
    )
    val_ds = OCTDLDataset(val_data, classes, transform=base_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = get_model_by_type(
        "MobileNetV2", True, classes, dropout)

    adam = optim.Adam(model.parameters(), learning_rate)

    train_gen = train(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        optimizer=adam,
        loss_fn=loss_fn,
        metrics=metrics,
        patience=5,
        from_epoch=20,
    )

    res = None
    for epoch_result in train_gen:
        res = epoch_result
        print(epoch_result.val_loss)
        print(epoch_result.val_metrics)

    print(json.dumps(res.val_confusion_matrix.tolist()))
