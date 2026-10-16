import numpy as np
from abc import ABC, abstractmethod
from typing import Any


def polynomial_feature_exprs(gene_exprs: list, poly: Any) -> list:
    """Reconstruct polynomial-feature expressions from per-gene expressions.

    Given the SymPy expression of each gene and a fitted
    ``PolynomialFeatures(include_bias=False)``, return one expression per
    output feature, in the same row order as ``poly.powers_`` — i.e.
    ``prod_j gene_exprs[j] ** powers[k, j]`` for feature ``k``. This lets a
    combiner rebuild its fitted model symbolically, cross-terms included.
    """
    feats = []
    for powers in poly.powers_:
        term = None
        for gi, p in enumerate(powers):
            if p:
                factor = gene_exprs[gi] ** int(p)
                term = factor if term is None else term * factor
        feats.append(term if term is not None else 1)
    return feats


class BaseCombiner(ABC):
    coef_: np.ndarray
    bias_: float
    gene_weights_: np.ndarray

    @abstractmethod
    def fit(self, G: np.ndarray, y: np.ndarray) -> "BaseCombiner": ...

    @abstractmethod
    def predict(self, G: np.ndarray) -> np.ndarray: ...

    def symbolic(self, gene_exprs: list) -> Any:
        """Return the fitted model as a single SymPy expression.

        Combines the per-gene SymPy expressions exactly as the combiner
        combines gene outputs numerically (bias, coefficients, polynomial
        expansion, and any feature scaling). Used by
        :meth:`~symgene.results.PopulationResult.to_sympy`.
        """
        raise NotImplementedError
