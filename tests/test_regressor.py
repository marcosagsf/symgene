import numpy as np
import pytest
from symgene.regressor import SymGeneRegressor

rng = np.random.default_rng(1)
X = rng.standard_normal((60, 4))
y = X[:, 0] ** 2 + X[:, 1] - X[:, 2] * X[:, 3]

def test_regressor_fit_predict():
    model = SymGeneRegressor(n_genes=3, n_gen=3, pop_size=15, seed=0)
    model.fit(X, y)
    y_pred = model.predict(X)
    assert y_pred.shape == (60,)

def test_regressor_best_expression_is_string():
    model = SymGeneRegressor(n_genes=3, n_gen=3, pop_size=15, seed=0)
    model.fit(X, y)
    assert isinstance(model.best_expression_, str)

def test_regressor_to_callable():
    model = SymGeneRegressor(n_genes=3, n_gen=3, pop_size=15, seed=0)
    model.fit(X, y)
    fn = model.to_callable()
    assert fn is not None
    assert fn(X).shape == (60,)

def test_regressor_with_validation():
    X_val = rng.standard_normal((20, 4))
    y_val = X_val[:, 0] ** 2 + X_val[:, 1] - X_val[:, 2] * X_val[:, 3]
    model = SymGeneRegressor(n_genes=3, n_gen=3, pop_size=15, seed=0)
    model.fit(X, y, X_val=X_val, y_val=y_val)
    assert "val_r2" in model._result.history_[0]

def test_regressor_with_feature_names():
    model = SymGeneRegressor(
        n_genes=3, n_gen=3, pop_size=15, seed=0,
        feature_names=["age", "bmd", "weight", "height"],
    )
    model.fit(X, y)
    expr = model.best_expression_
    assert any(name in expr for name in ["age", "bmd", "weight", "height"])


# ── score() method ────────────────────────────────────────────────────────────

def test_score_returns_float():
    model = SymGeneRegressor(n_genes=2, pop_size=10, n_gen=3, seed=0, verbose=0)
    model.fit(X, y)
    result = model.score(X, y)
    assert isinstance(result, float)


def test_score_matches_r2_formula():
    model = SymGeneRegressor(n_genes=2, pop_size=10, n_gen=3, seed=0, verbose=0)
    model.fit(X, y)
    y_pred = model.predict(X)
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    expected = 1.0 - ss_res / ss_tot
    assert model.score(X, y) == pytest.approx(expected)


def test_score_not_fitted_raises():
    model = SymGeneRegressor(n_genes=2, pop_size=8, n_gen=1, verbose=0)
    with pytest.raises((AttributeError, TypeError)):
        model.score(X, y)
