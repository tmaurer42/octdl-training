import abc

from sklearn import metrics


class CategoricalMetric(abc.ABC):
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
        return sum(self.collected_values) / len(self.collected_values)
    

    def reset(self):
        self.collected_values = []
    

class BalancedAccuracy(CategoricalMetric):
    def __init__(self):
        super().__init__()

    def calculate_metric(self, predictions: list[int], targets: list[int]) -> float:
        return metrics.balanced_accuracy_score(targets, predictions)
    

class F1Score(CategoricalMetric):
    def __init__(self):
        super().__init__()

    def calculate_metric(self, predictions: list[int], targets: list[int]) -> float:
        return metrics.f1_score(targets, predictions, average="macro")