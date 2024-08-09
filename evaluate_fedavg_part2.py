from evaluate import FLEvalParameters, evaluate
from shared.data import OCTDLClass

def main():
    classes = [OCTDLClass.AMD, OCTDLClass.NO]

    n_epochs_to_try = [5, 10]
    n_clients_to_try = [20]

    for n_clients in n_clients_to_try:
        for n_epochs in n_epochs_to_try:
            evaluate(
                classes=classes,
                model_type='ResNet18',
                transfer_learning=False,
                optimization_mode='minimize_loss',
                loss_fn_type='CrossEntropy',
                fl_eval_parameters=FLEvalParameters(
                    strategy='FedAvg',
                    n_clients=n_clients,
                    n_epochs=n_epochs
                )
            )
            evaluate(
                classes=classes,
                model_type='ResNet18',
                transfer_learning=True,
                optimization_mode='minimize_loss',
                loss_fn_type='CrossEntropy',
                fl_eval_parameters=FLEvalParameters(
                    strategy='FedAvg',
                    n_clients=n_clients,
                    n_epochs=n_epochs
                )
            )
            evaluate(
                classes=classes,
                model_type='MobileNetV2',
                transfer_learning=True,
                optimization_mode='minimize_loss',
                loss_fn_type='CrossEntropy',
                fl_eval_parameters=FLEvalParameters(
                    strategy='FedAvg',
                    n_clients=n_clients,
                    n_epochs=n_epochs
                )
            )

if __name__ == "__main__":
    main()