from centralized.optimization import main as run_centralized
from federated_learning.optimization import main as run_federated
from shared.data import OCTDLClass


def main():
    classes = [OCTDLClass.AMD, OCTDLClass.DME, OCTDLClass.ERM,
               OCTDLClass.NO, OCTDLClass.RAO, OCTDLClass.RVO, OCTDLClass.VID]

    #run_centralized(model_type='ResNet50', transfer_learning=False, loss_fn_type='WeightedCrossEntropy',
    #                class_list=classes, optimization_mode='maximize_f1_macro', n_jobs=1)
    #run_centralized(model_type='MobileNetV2', transfer_learning=True, loss_fn_type='WeightedCrossEntropy',
    #                class_list=classes, optimization_mode='maximize_f1_macro', n_jobs=1)

    n_clients = 20
    n_total_updates = 260

    for n_clients_per_round in [10, 5, 3]:
        n_rounds = n_total_updates // n_clients_per_round
        run_federated(model_type="ResNet50", transfer_learning=False, loss_fn_type='WeightedCrossEntropy',
                      class_list=classes, optimization_mode='maximize_f1_macro', n_jobs=1,
                      fl_strategy='FedAvg', n_clients=n_clients, n_rounds=n_rounds,
                      n_clients_per_round=n_clients_per_round)
        run_federated(model_type="MobileNetV2", transfer_learning=True, loss_fn_type='WeightedCrossEntropy',
                      class_list=classes, optimization_mode='maximize_f1_macro', n_jobs=1,
                      fl_strategy='FedAvg', n_clients=n_clients, n_rounds=n_rounds,
                      n_clients_per_round=n_clients_per_round)

    for n_clients_per_round in [10, 5, 3]:
        n_rounds = n_total_updates // n_clients_per_round
        run_federated(model_type="ResNet50", transfer_learning=False, loss_fn_type='WeightedCrossEntropy',
                      class_list=classes, optimization_mode='maximize_f1_macro', n_jobs=1,
                      fl_strategy='FedBuff', n_clients=n_clients, n_rounds=n_rounds,
                      n_clients_per_round=n_clients_per_round)
        run_federated(model_type="MobileNetV2", transfer_learning=True, loss_fn_type='WeightedCrossEntropy',
                      class_list=classes, optimization_mode='maximize_f1_macro', n_jobs=1,
                      fl_strategy='FedBuff', n_clients=n_clients, n_rounds=n_rounds,
                      n_clients_per_round=n_clients_per_round)


if __name__ == "__main__":
    main()
