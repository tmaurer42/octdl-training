from shared.data import OCTDLClass
from evaluate import evaluate


def main():
    evaluate([OCTDLClass.AMD, OCTDLClass.NO], 'ResNet18', transfer_learning=True,
             optimization_mode='minimize_loss', loss_fn_type='CrossEntropy')
    evaluate([OCTDLClass.AMD, OCTDLClass.NO], 'ResNet18', transfer_learning=False,
             optimization_mode='minimize_loss', loss_fn_type='CrossEntropy')
    evaluate([OCTDLClass.AMD, OCTDLClass.NO], 'MobileNetV2', transfer_learning=True,
             optimization_mode='minimize_loss', loss_fn_type='CrossEntropy')
    evaluate([OCTDLClass.AMD, OCTDLClass.NO], 'EfficientNetV2', transfer_learning=True,
             optimization_mode='minimize_loss', loss_fn_type='CrossEntropy')


if __name__ == "__main__":
    main()