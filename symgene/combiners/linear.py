import numpy as np
from sklearn.linear_model import LinearRegression
from symgene.combiners.base import BaseCombiner

class LinearCombiner(BaseCombiner):
    def __init__(self):
        self._model = LinearRegression()

    def fit(self, G: np.ndarray, y: np.ndarray) -> "LinearCombiner":
        self._model.fit(G, y)
        self.coef_ = self._model.coef_.flatten()
        self.bias_ = float(self._model.intercept_)
        self.gene_weights_ = np.abs(self.coef_)
        return self

    def predict(self, G: np.ndarray) -> np.ndarray:
        return self._model.predict(G)
