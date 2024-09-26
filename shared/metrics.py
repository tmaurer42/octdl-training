import abc
from sklearn import metrics


def balanced_accuracy(predictions, actual_values):
    recalls = metrics.recall_score(
        y_true=actual_values,
        y_pred=predictions,
        average=None
    )

    balanced_acc = recalls.mean()
    return balanced_acc


def f1_score_macro(predictions, actual_values):
    return metrics.f1_score(
        y_true=actual_values,
        y_pred=predictions,
        average='macro'
    )


class CategoricalMetric(abc.ABC):
    """
    Base class similar to torchmetrics to wrap sklearn metrics.
    """
    all_preds: list[int]
    all_targets: list[int]

    @staticmethod
    @abc.abstractmethod
    def name() -> str:
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

    @staticmethod
    def name() -> str:
        return "balanced_accuracy"

    def calculate_metric(self) -> float:
        return balanced_accuracy(
            predictions=self.all_preds,
            actual_values=self.all_targets
        )


class F1ScoreMacro(CategoricalMetric):
    """
    Metric class to compute the F1 score with macro average from sklearn
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def name() -> str:
        return "f1_score_macro"

    def calculate_metric(self) -> float:
        return f1_score_macro(
            predictions=self.all_preds,
            actual_values=self.all_targets
        )