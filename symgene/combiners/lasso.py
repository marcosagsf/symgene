import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import PolynomialFeatures
from symgene.combiners.base import BaseCombiner

class LassoCombiner(BaseCombiner):
    def __init__(self, alphas: list[float] | None = None, degree: int = 1):
        self.alphas = alphas if alphas is not None else [0.01, 0.1, 1.0]
        self.degree = degree
        self._poly: PolynomialFeatures | None = None
        self._model: LassoCV | None = None

    def _featurize(self, G: np.ndarray, fit: bool = False) -> np.ndarray:
        if self.degree > 1:
            if fit:
                self._poly = PolynomialFeatures(degree=self.degree, include_bias=False)
                return self._poly.fit_transform(G)
            return self._poly.transform(G)
        return G

    def fit(self, G: np.ndarray, y: np.ndarray) -> "LassoCombiner":
        features = self._featurize(G, fit=True)
        self._model = LassoCV(alphas=self.alphas, cv=5, max_iter=5000)
        self._model.fit(features, y)
        self.coef_ = self._model.coef_.flatten()
        self.bias_ = float(self._model.intercept_)
        self.gene_weights_ = np.abs(self.coef_[:G.shape[1]])
        return self

    def predict(self, G: np.ndarray) -> np.ndarray:
        return self._model.predict(self._featurize(G))
