from abc import ABC, abstractmethod
from typing import Callable, Sequence
from dataclasses import dataclass
import numpy as np


@dataclass
class OptimResult:
    x_best: np.ndarray        # best solution found, shape (n_dims,)
    f_best: float             # objective value at x_best
    history: list[float]      # best f per iteration
    n_eval: int               # total function evaluations


class Optimizer(ABC):
    @abstractmethod
    def optimize(
        self,
        fn: Callable[[np.ndarray], float],
        bounds: Sequence[tuple[float, float]],
        seed: int | None = None,
    ) -> OptimResult:
        """Minimize fn within bounds. Returns OptimResult."""
