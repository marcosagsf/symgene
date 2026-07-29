import numpy as np
from abc import ABC, abstractmethod

class BaseCombiner(ABC):
    coef_: np.ndarray
    bias_: float
    gene_weights_: np.ndarray

    @abstractmethod
    def fit(self, G: np.ndarray, y: np.ndarray) -> "BaseCombiner": ...

    @abstractmethod
    def predict(self, G: np.ndarray) -> np.ndarray: ...
