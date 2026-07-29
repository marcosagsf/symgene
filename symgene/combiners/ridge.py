import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import PolynomialFeatures
from symgene.combiners.base import BaseCombiner

class RidgeCombiner(BaseCombiner):
    def __init__(self, alphas: list[float] | None = None, degree: int = 1):
        self.alphas = alphas if alphas is not None else [0.1, 1.0, 10.0, 100.0]
        self.degree = degree
        self._poly: PolynomialFeatures | None = None
        self._model: RidgeCV | None = None

    def _featurize(self, G: np.ndarray, fit: bool = False) -> np.ndarray:
        if self.degree > 1:
            if fit:
                self._poly = PolynomialFeatures(degree=self.degree, include_bias=False)
                return self._poly.fit_transform(G)
            return self._poly.transform(G)
        return G

    def fit(self, G: np.ndarray, y: np.ndarray) -> "RidgeCombiner":
        features = self._featurize(G, fit=True)
        self._model = RidgeCV(alphas=self.alphas, cv=None)
        self._model.fit(features, y)
        self.coef_ = self._model.coef_.flatten()
        self.bias_ = float(self._model.intercept_)
        self._compute_gene_weights(G.shape[1])
        return self

    def predict(self, G: np.ndarray) -> np.ndarray:
        return self._model.predict(self._featurize(G))

    def _compute_gene_weights(self, n_genes: int):
        if self.degree > 1 and self._poly is not None:
            weights = np.zeros(n_genes)
            for gi in range(n_genes):
                mask = self._poly.powers_[:, gi] > 0
                weights[gi] = np.abs(self.coef_[mask]).sum()
            self.gene_weights_ = weights
        else:
            self.gene_weights_ = np.abs(self.coef_)
