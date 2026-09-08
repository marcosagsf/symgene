import numpy as np
from typing import Any, Callable

class FitnessEvaluator:
    def __init__(
        self,
        metric: Callable,
        penalties: list[Callable] | None = None,
    ) -> None:
        self.metric = metric
        self.penalties = penalties if penalties is not None else []

    def compute(self, individual: Any, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray) -> float:
        score = self.metric(y, y_pred)
        for pen_fn in self.penalties:
            score += pen_fn(individual, X, y)
        return float(score)
