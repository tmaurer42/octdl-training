from federated_learning.optimization import main as run_experiment
from shared.data import OCTDLClass


def run_fedbuff_experiments():
    n_jobs = 1
    classes = [OCTDLClass.AMD, OCTDLClass.NO]
    n_clients = 20
    n_total_updates = 260

    for buffer_size in [10, 5, 3]:
        for model_type, transfer_learning in [('ResNet18', True), ('ResNet18', False), ('MobileNetV2', True)]:
            n_rounds = n_total_updates // buffer_size
            try:
                run_experiment(model_type=model_type, transfer_learning=transfer_learning, loss_fn_type='WeightedCrossEntropy',
                        class_list=classes, optimization_mode='maximize_f1_macro', n_jobs=n_jobs,
                        fl_strategy='FedBuff', n_clients=n_clients, n_rounds=n_rounds,
                        n_clients_per_round=buffer_size)
            except:
                print("An experiment crashed :(")

if __name__ == "__main__":
    run_fedbuff_experiments()