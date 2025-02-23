from shared.data import OCTDLClass
from evaluate import evaluate


def main():
    evaluate([OCTDLClass.AMD, OCTDLClass.NO], 'ResNet18', transfer_learning=True,
             optimization_mode='maximize_f1_macro', loss_fn_type='WeightedCrossEntropy')
    evaluate([OCTDLClass.AMD, OCTDLClass.NO], 'ResNet18', transfer_learning=False,
             optimization_mode='maximize_f1_macro', loss_fn_type='WeightedCrossEntropy')
    evaluate([OCTDLClass.AMD, OCTDLClass.NO], 'MobileNetV2', transfer_learning=True,
             optimization_mode='maximize_f1_macro', loss_fn_type='WeightedCrossEntropy')


if __name__ == "__main__":
    main()