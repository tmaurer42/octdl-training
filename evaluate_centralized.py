import os

import optuna
import torch
from torch import nn
import torch.utils
import torch.utils.data
from sklearn import metrics

from experiments_centralized import get_study_name, load_weights, OptimizationMode
from shared.data import OCTDLClass, OCTDLDataset, get_transforms, load_octdl_data
from shared.metrics import balanced_accuracy, balanced_accuracy_from_confustion_matrix
from shared.model import ModelType, get_model_by_type
from train_centralized import set_device


def get_result_db_name(
    model_type: ModelType,
    classes: list[OCTDLClass],
    optimization_mode: OptimizationMode
):
    classes_str = '-'.join([cls.name for cls in classes])
    return f"results_{model_type}_{classes_str}_{optimization_mode}.sqlite3"


def get_study(
    classes: list[OCTDLClass],
    model_type: ModelType,
    transfer_learning: bool,
    loss_fn: nn.CrossEntropyLoss,
    optimization_mode: OptimizationMode
):
    study_name = get_study_name(
        classes,
        model_type,
        transfer_learning=transfer_learning,
        loss=loss_fn,
        optimization_mode=optimization_mode
    )

    db_name = get_result_db_name(model_type, classes, optimization_mode)
    db_url = f"sqlite:///{os.path.join('results_centralized', db_name)}"
    study: optuna.Study = optuna.load_study(
        study_name=study_name, storage=db_url)

    return study


def get_study_with_best_acc(studies: list[optuna.Study]):
    best_acc = 0.0
    best_study: optuna.Study = None

    for study in studies:
        best_trial = study.best_trial

        cm = best_trial.user_attrs['confusion_matrix']
        # Choos better model by F1 Score
        val_bal_acc = balanced_accuracy_from_confustion_matrix(cm)

        if val_bal_acc > best_acc:
            best_acc = val_bal_acc
            best_study = study

    return best_study


def evaluate(
    classes: list[OCTDLClass],
    model_type: ModelType,
    transfer_learning: bool,
    optimization_mode: OptimizationMode
):
    ce_loss = nn.CrossEntropyLoss()
    weighted_ce_loss = nn.CrossEntropyLoss(weight=torch.tensor([]))

    study_ce_loss = get_study(
        classes, model_type, transfer_learning, ce_loss, optimization_mode)
    study_weighted_ce_loss = get_study(
        classes, model_type, transfer_learning, weighted_ce_loss, optimization_mode)

    best_study = get_study_with_best_acc(
        [study_ce_loss, study_weighted_ce_loss])
    study_name = best_study.study_name
    best_trial = best_study.best_trial

    _, _, test_data = load_octdl_data(classes)
    base_transform, _ = get_transforms(224)
    test_dataset = OCTDLDataset(test_data, classes, base_transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False)

    model = get_model_by_type(model_type, transfer_learning, classes, 0.0)
    load_weights(model, study_name, best_trial.number)

    all_preds = []
    all_labels = []
    device = set_device()
    model.eval()
    model.to(device)
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs.data, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    bal_acc = balanced_accuracy(all_preds, all_labels)
    f1score_macro = metrics.f1_score(all_labels, all_preds, average='macro')
    f1score_amd = metrics.f1_score(
        all_labels, all_preds, average='binary', pos_label=0)

    classes_str = ','.join([cls.name for cls in classes])
    print()
    print(
        f"-- Evaluation on testset for [{classes_str}] using model {model_type}",
        f"with{'' if transfer_learning else 'out'} transfer learning --")
    print(f"-- Study name: {best_study.study_name} --")
    print(f"Balanced Accuracy: {bal_acc}")
    print(f"F1 Score averaged: {f1score_macro}")
    print(f"F1 Score for AMD: {f1score_amd}")
    print()


def main():
    evaluate([OCTDLClass.AMD, OCTDLClass.NO],
             'ResNet18', transfer_learning=False, optimization_mode='minimize_loss')
    evaluate([OCTDLClass.AMD, OCTDLClass.NO],
             'ResNet18', transfer_learning=True, optimization_mode='minimize_loss')
    evaluate([OCTDLClass.AMD, OCTDLClass.NO],
             'MobileNetV2', transfer_learning=True, optimization_mode='minimize_loss')
    evaluate([OCTDLClass.AMD, OCTDLClass.NO],
             'EfficientNetV2', transfer_learning=True, optimization_mode='minimize_loss')


if __name__ == "__main__":
    main()
