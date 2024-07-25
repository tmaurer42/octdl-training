import abc
from sklearn import metrics


def balanced_accuracy(predictions, actual_values):
    recalls = metrics.recall_score(actual_values, predictions, average=None)
    
    balanced_acc = recalls.mean()
    return balanced_acc

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
    collected_values: list[float]

    def __init__(self):
        self.collected_values = []

    @abc.abstractmethod
    def calculate_metric(self, predictions: list[int], targets: list[int]) -> float:
        pass

    def update(self, predictions: list[int], targets: list[int]):
        value = self.calculate_metric(predictions, targets)
        self.collected_values.append(value)

        return value

    def compute(self):
        if len(self.collected_values) == 0:
            return 0.0
        return sum(self.collected_values) / len(self.collected_values)

    def reset(self):
        self.collected_values = []


class BalancedAccuracy(CategoricalMetric):
    """
    Metric class to compute the balanced accuarcy score from sklearn
    """

    def __init__(self):
        super().__init__()

    def calculate_metric(self, predictions: list[int], targets: list[int]) -> float:
        return metrics.balanced_accuracy_score(targets, predictions)


class F1ScoreMacro(CategoricalMetric):
    """
    Metric class to compute the F1 score with macro average from sklearn
    """

    def __init__(self):
        super().__init__()

    def calculate_metric(self, predictions: list[int], targets: list[int]) -> float:
        return metrics.f1_score(targets, predictions, average="macro")
