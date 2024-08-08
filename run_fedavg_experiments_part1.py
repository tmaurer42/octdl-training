from federated_learning.optimization import main as run_experiment
from shared.data import OCTDLClass


def run_fedavg_experiments():
    n_jobs = 1
    classes = [OCTDLClass.AMD, OCTDLClass.NO]
    n_rounds = 10

    n_epochs_to_try = [1, 5]
    n_clients_to_try = [10]

    for n_clients in n_clients_to_try:
        for n_epochs in n_epochs_to_try:
            run_experiment(model_type='ResNet18', transfer_learning=True, loss_fn_type='CrossEntropy',
                           class_list=classes, optimization_mode='minimize_loss', n_jobs=n_jobs,
                           fl_strategy='FedAvg', n_clients=n_clients, n_local_epochs=n_epochs, n_rounds=n_rounds)

            run_experiment(model_type='ResNet18', transfer_learning=False, loss_fn_type='CrossEntropy',
                           class_list=classes, optimization_mode='minimize_loss', n_jobs=n_jobs,
                           fl_strategy='FedAvg', n_clients=n_clients, n_local_epochs=n_epochs, n_rounds=n_rounds)
            
            run_experiment(model_type='MobileNetV2', transfer_learning=True, loss_fn_type='CrossEntropy',
                           class_list=classes, optimization_mode='minimize_loss', n_jobs=n_jobs,
                           fl_strategy='FedAvg', n_clients=n_clients, n_local_epochs=n_epochs, n_rounds=n_rounds)

            run_experiment(model_type='EfficientNetV2', transfer_learning=True, loss_fn_type='CrossEntropy',
                           class_list=classes, optimization_mode='minimize_loss', n_jobs=n_jobs,
                           fl_strategy='FedAvg', n_clients=n_clients, n_local_epochs=n_epochs, n_rounds=n_rounds)


if __name__ == "__main__":
    run_fedavg_experiments()
