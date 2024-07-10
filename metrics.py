import abc
from typing import List

from sklearn import metrics


class CategoricalMetric(abc.ABC):
    collected_values: List[float]

    def __init__(self):
        self.collected_values = []

    @abc.abstractmethod
    def calculate_metric(self, predictions: List[int], targets: List[int]) -> float:
        pass

    def update(self, predictions: List[int], targets: List[int]):
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

    def calculate_metric(self, predictions: List[int], targets: List[int]) -> float:
        return metrics.balanced_accuracy_score(targets, predictions)
    

metric = BalancedAccuracy()

metric.update([1, 0, 0], [1, 0, 0])
metric.update([1, 0, 0], [1, 0, 1])

print(metric.compute())