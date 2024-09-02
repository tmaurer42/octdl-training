from evaluate import FLEvalParameters, evaluate
from shared.data import OCTDLClass


def eval_fedbuff_experiments():
    classes = [OCTDLClass.AMD, OCTDLClass.NO]
    n_clients = 20

    for buffer_size in [10, 5, 3]:
        fl_eval_params = FLEvalParameters(
            strategy='FedBuff', n_clients=n_clients, n_clients_per_round=buffer_size)
        evaluate(
            classes=classes,
            model_type="ResNet18",
            transfer_learning=True,
            optimization_mode='maximize_f1_macro',
            loss_fn_type='WeightedCrossEntropy',
            fl_eval_parameters=fl_eval_params
        )
        evaluate(
            classes=classes,
            model_type="ResNet18",
            transfer_learning=False,
            optimization_mode='maximize_f1_macro',
            loss_fn_type='WeightedCrossEntropy',
            fl_eval_parameters=fl_eval_params
        )
        evaluate(
            classes=classes,
            model_type="MobileNetV2",
            transfer_learning=True,
            optimization_mode='maximize_f1_macro',
            loss_fn_type='WeightedCrossEntropy',
            fl_eval_parameters=fl_eval_params
        )


if __name__ == "__main__":
    eval_fedbuff_experiments()
