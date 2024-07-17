from torch import nn, optim
from torch.utils.data import DataLoader

from shared.data import OCTDLClass, OCTDLDataset, get_balancing_weights, load_octdl_data, get_transforms
from shared.metrics import BalancedAccuracy, F1ScoreMacro
from shared.model import get_efficientnet
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
    metric_names = ['balanced_accuracy', 'f1_score']
    loss_fn = nn.CrossEntropyLoss()

    image_size = 224
    epochs = 100

    batch_size = 32
    learning_rate = 0.0005
    apply_augmentation = True
    dropout = 0.1

    base_transform, train_transform = get_transforms(image_size)
    train_ds = OCTDLDataset(
        train_data,
        classes,
        transform=train_transform if apply_augmentation else base_transform
    )
    val_ds = OCTDLDataset(val_data, classes, transform=base_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = get_efficientnet(
        transfer_learning=False,
        num_classes=len(classes),
        dropout=dropout
    )

    adam = optim.Adam(model.parameters(), learning_rate)

    train_gen = train(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        optimizer=adam,
        loss_fn=loss_fn,
        metrics=metrics,
        metric_names=metric_names,
        patience=5,
        from_epoch=20
    )

    while True:
        try:
            running_loss = next(train_gen)
        except StopIteration as res:
            best_val_loss, best_confusion_matrix, best_model_metrics = res.value
            print(best_val_loss)
