from federated_learning.optimization import main as run_experiment
from shared.data import OCTDLClass


def run_fedavg_experiments():
    n_jobs = 1
    classes = [OCTDLClass.AMD, OCTDLClass.NO]
    n_clients = 20
    buffer_size = 10
    n_rounds = 40
            
    run_experiment(model_type='MobileNetV2', transfer_learning=True, loss_fn_type='WeightedCrossEntropy',
                class_list=classes, optimization_mode='minimize_loss', n_jobs=n_jobs,
                fl_strategy='FedAvg', n_clients=n_clients, n_rounds=n_rounds,
                n_clients_per_round=buffer_size)


if __name__ == "__main__":
    run_fedavg_experiments()