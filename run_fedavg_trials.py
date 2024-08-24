from federated_learning.optimization import main as run_experiment
from shared.data import OCTDLClass


def run_fedavg_experiments():
    n_jobs = 1
    classes = [OCTDLClass.AMD, OCTDLClass.NO]
    n_clients = 20
    n_total_updates = 260

    for clients_per_round in [10, 5, 3]:
        n_rounds = n_total_updates // clients_per_round
        run_experiment(model_type='ResNet18', transfer_learning=True, loss_fn_type='WeightedCrossEntropy',
                    class_list=classes, optimization_mode='maximize_f1_macro', n_jobs=n_jobs,
                    fl_strategy='FedAvg', n_clients=n_clients, n_rounds=n_rounds,
                    n_clients_per_round=clients_per_round)
        run_experiment(model_type='ResNet18', transfer_learning=False, loss_fn_type='WeightedCrossEntropy',
                    class_list=classes, optimization_mode='maximize_f1_macro', n_jobs=n_jobs,
                    fl_strategy='FedAvg', n_clients=n_clients, n_rounds=n_rounds,
                    n_clients_per_round=clients_per_round)
        run_experiment(model_type='MobileNetV2', transfer_learning=True, loss_fn_type='WeightedCrossEntropy',
                    class_list=classes, optimization_mode='maximize_f1_macro', n_jobs=n_jobs,
                    fl_strategy='FedAvg', n_clients=n_clients, n_rounds=n_rounds,
                    n_clients_per_round=clients_per_round)


if __name__ == "__main__":
    run_fedavg_experiments()