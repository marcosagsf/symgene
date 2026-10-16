"""The symbolic export must equal the numerical model, including the combiner.

Covers the fix where to_sympy/to_latex previously ignored the combiner
(coefficients, bias, and polynomial cross-terms), and the ARG0 feature-name
bug in terminal rendering.
"""
import warnings

import numpy as np
import sympy as sp
import pytest

from symgene import SymGeneRegressor

# Primitives that lambdify cleanly to NumPy, so we can evaluate the symbolic
# model and compare it to predict().
PRIMS = ["add", "sub", "mul", "sin", "cos", "square"]


def _fit(combiner, degree, seed=1):
    rng = np.random.default_rng(0)
    X = rng.uniform(-1.2, 1.2, (160, 2))
    y = X[:, 0] ** 2 + X[:, 0] * X[:, 1] - X[:, 1]
    reg = SymGeneRegressor(
        n_genes=2, pop_size=40, n_gen=12, combiner=combiner,
        regression_degree=degree, primitives=PRIMS, squash=None,
        feature_names=["x1", "x2"], seed=seed, verbose=0,
    )
    reg.fit(X, y)
    return reg


@pytest.mark.parametrize("combiner,degree", [
    ("linear", 1),
    ("ridge", 1),
    ("ridge", 2),
    ("lasso", 2),
])
def test_to_sympy_matches_predict(combiner, degree):
    reg = _fit(combiner, degree)
    expr = reg._result.to_sympy()
    assert expr is not None
    f = sp.lambdify((sp.Symbol("x1"), sp.Symbol("x2")), expr, "numpy")
    Xc = np.random.default_rng(7).uniform(-1.2, 1.2, (15, 2))
    y_sym = np.array([float(f(a, b)) for a, b in Xc])
    y_prd = reg.predict(Xc)
    np.testing.assert_allclose(y_sym, y_prd, rtol=1e-6, atol=1e-6)


def test_gene_weights_aligned_with_genes_degree2():
    reg = _fit("ridge", 2)
    r = reg._result
    assert r.gene_weights_ is not None
    assert len(r.gene_weights_) == r.n_genes_          # per-gene, not poly-space
    assert len(r.coefficients_) >= r.n_genes_          # raw coef lives in poly space


def test_feature_names_render_not_arg():
    reg = _fit("ridge", 1)
    latex = reg._result.to_latex()
    assert "ARG" not in latex
    assert "x_{1}" in latex or "x1" in latex


def test_to_latex_is_string():
    reg = _fit("linear", 1)
    assert isinstance(reg._result.to_latex(), str)


def test_gene_sympy_returns_per_gene_list():
    reg = _fit("ridge", 1)
    genes = reg._result.gene_sympy()
    assert isinstance(genes, list)
    assert len(genes) == reg._result.n_genes_
