import abc
from sklearn import metrics


def balanced_accuracy(predictions, actual_values):
    recalls = metrics.recall_score(actual_values, predictions, average=None)

    balanced_acc = recalls.mean()
    return balanced_acc


def f1_score_macro(predictions, actual_values):
    return metrics.f1_score(actual_values, predictions, average='macro')


def balanced_accuracy_from_confustion_matrix(cm):
    recalls = []
    for i in range(len(cm)):
        tp = cm[i][i]
        fn = sum(cm[i]) - tp
        recall = tp / (tp + fn)
        recalls.append(recall)

    balanced_acc = sum(recalls) / len(recalls)
    return balanced_acc


class CategoricalMetric(abc.ABC):
    """
    Base class similar to torchmetrics to wrap sklearn metrics.
    """
    all_preds: list[int]
    all_targets: list[int]

    @property
    @abc.abstractmethod
    def name(self) -> str:
        return None 

    def __init__(self):
        self.all_preds = []
        self.all_targets = []

    @abc.abstractmethod
    def calculate_metric(self) -> float:
        pass

    def update(self, predictions: list[int], targets: list[int]):
        self.all_preds.extend(predictions)
        self.all_targets.extend(targets)

        return self.calculate_metric()

    def compute(self):
        return self.calculate_metric()

    def reset(self):
        self.all_preds = []
        self.all_targets = []


class BalancedAccuracy(CategoricalMetric):
    """
    Metric class to compute the balanced accuarcy score from sklearn
    """

    def __init__(self):
        super().__init__()
        self._name = "balanced_accuracy"

    @property
    def name(self) -> str:
        return self._name

    def calculate_metric(self) -> float:
        return balanced_accuracy(self.all_targets, self.all_preds)


class F1ScoreMacro(CategoricalMetric):
    """
    Metric class to compute the F1 score with macro average from sklearn
    """

    def __init__(self):
        super().__init__()
        self._name = "f1_score_macro"

    @property
    def name(self) -> str:
        return self._name

    def calculate_metric(self) -> float:
        return f1_score_macro(self.all_targets, self.all_preds)
