from evaluate import FLEvalParameters, evaluate
from shared.data import OCTDLClass


def eval_resnet50_trials():
    classes = [OCTDLClass.AMD, OCTDLClass.NO]
    n_clients = 20

    evaluate([OCTDLClass.AMD, OCTDLClass.NO], 'ResNet50', transfer_learning=False,
            optimization_mode='maximize_f1_macro', loss_fn_type='WeightedCrossEntropy')

    for n_clients_per_round in [10, 5, 3]:
        fl_eval_params = FLEvalParameters(
            strategy='FedAvg', n_clients=n_clients, n_clients_per_round=n_clients_per_round)
        evaluate(
            classes=classes,
            model_type="ResNet50",
            transfer_learning=False,
            optimization_mode='maximize_f1_macro',
            loss_fn_type='WeightedCrossEntropy',
            fl_eval_parameters=fl_eval_params
        )

    for buffer_size in [10, 5, 3]:
        fl_eval_params = FLEvalParameters(
            strategy='FedBuff', n_clients=n_clients, n_clients_per_round=buffer_size)
        evaluate(
            classes=classes,
            model_type="ResNet50",
            transfer_learning=False,
            optimization_mode='maximize_f1_macro',
            loss_fn_type='WeightedCrossEntropy',
            fl_eval_parameters=fl_eval_params
        )