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
