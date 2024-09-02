from centralized.optimization import main as run_experiment
from shared.data import OCTDLClass


def main():
    classes = [OCTDLClass.AMD, OCTDLClass.NO]

    run_experiment(model_type='ResNet18', transfer_learning=True, loss_fn_type='WeightedCrossEntropy',
                    class_list=classes, optimization_mode='maximize_f1_macro', n_jobs=2)

    run_experiment(model_type='ResNet18', transfer_learning=False, loss_fn_type='WeightedCrossEntropy',
                    class_list=classes, optimization_mode='maximize_f1_macro', n_jobs=2)

    run_experiment(model_type='MobileNetV2', transfer_learning=True, loss_fn_type='WeightedCrossEntropy',
                    class_list=classes, optimization_mode='maximize_f1_macro', n_jobs=2)
                    
    #run_experiment(model_type='EfficientNetV2', transfer_learning=True, loss_fn_type='WeightedCrossEntropy',
    #                class_list=classes, optimization_mode='maximize_f1_macro', n_jobs=2)


if __name__ == "__main__":
    main()
