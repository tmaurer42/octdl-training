from evaluate import FLEvalParameters, evaluate
from shared.data import OCTDLClass


def eval_fedbuff_experiments():
    classes = [OCTDLClass.AMD, OCTDLClass.NO]

    buffer_sizes_to_try = [3, 5, 10]
    n_clients_to_try = [20]

    for n_clients in n_clients_to_try:
        for buffer_size in buffer_sizes_to_try:
            evaluate(
                classes=classes, 
                model_type='ResNet18', 
                transfer_learning=True, 
                optimization_mode='minimize_loss', 
                loss_fn_type='CrossEntropy', 
                fl_eval_parameters=FLEvalParameters(
                    strategy='FedBuff',
                    buffer_size=buffer_size,
                    n_clients=n_clients,
                )
            )

            evaluate(
                classes=classes, 
                model_type='ResNet18', 
                transfer_learning=False, 
                optimization_mode='minimize_loss', 
                loss_fn_type='CrossEntropy', 
                fl_eval_parameters=FLEvalParameters(
                    strategy='FedBuff',
                    buffer_size=buffer_size,
                    n_clients=n_clients,
                )
            )

            evaluate(
                classes=classes, 
                model_type='MobileNetV2', 
                transfer_learning=True, 
                optimization_mode='minimize_loss', 
                loss_fn_type='CrossEntropy', 
                fl_eval_parameters=FLEvalParameters(
                    strategy='FedBuff',
                    buffer_size=buffer_size,
                    n_clients=n_clients,
                )
            )


if __name__ == "__main__":
    eval_fedbuff_experiments()