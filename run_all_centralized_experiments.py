from experiments_centralized import OptimizationMode, main as run_experiment
from shared.data import OCTDLClass


def main():
    classes = [OCTDLClass.AMD, OCTDLClass.NO]
    optimization_modes: list[OptimizationMode] = [
        'minimize_loss', 'maximize_f1_macro']

    for mode in optimization_modes:
        run_experiment(model_type='ResNet18', transfer_learning=True,
                       class_list=classes, optimization_mode=mode, n_jobs=2)

        run_experiment(model_type='ResNet18', transfer_learning=False,
                       class_list=classes, optimization_mode=mode, n_jobs=2)

        run_experiment(model_type='MobileNetV2', transfer_learning=True,
                       class_list=classes, optimization_mode=mode, n_jobs=2)
                       
        run_experiment(model_type='EfficientNetV2', transfer_learning=True,
                       class_list=classes, optimization_mode=mode, n_jobs=2)


if __name__ == "__main__":
    main()
