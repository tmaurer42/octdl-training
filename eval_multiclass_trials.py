from evaluate import FLEvalParameters, evaluate
from shared.data import OCTDLClass


def eval_multiclass_trials():
    classes = [OCTDLClass.AMD, OCTDLClass.DME, OCTDLClass.ERM,
               OCTDLClass.NO, OCTDLClass.RAO, OCTDLClass.RVO, OCTDLClass.VID]
    n_clients = 20

    evaluate(classes, 'ResNet50', transfer_learning=False,
            optimization_mode='maximize_f1_macro', loss_fn_type='WeightedCrossEntropy')
    evaluate(classes, 'MobileNetV2', transfer_learning=True,
            optimization_mode='maximize_f1_macro', loss_fn_type='WeightedCrossEntropy')

if __name__ == "__main__":
    eval_multiclass_trials()