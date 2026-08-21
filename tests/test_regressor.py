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


# ── get_params() / set_params() ───────────────────────────────────────────────

def test_get_params_returns_all_constructor_keys():
    model = SymGeneRegressor(n_genes=5, pop_size=20, n_gen=10, seed=7, verbose=0)
    params = model.get_params()
    for key in ("n_genes", "pop_size", "n_gen", "primitives", "squash",
                "combiner", "ridge_alphas", "regression_degree",
                "feature_names", "seed", "verbose"):
        assert key in params, f"Missing key: {key}"


def test_get_params_values_match_constructor():
    model = SymGeneRegressor(
        n_genes=5, pop_size=20, n_gen=10, combiner="ridge",
        regression_degree=2, seed=99, verbose=0,
    )
    p = model.get_params()
    assert p["n_genes"] == 5
    assert p["pop_size"] == 20
    assert p["n_gen"] == 10
    assert p["regression_degree"] == 2
    assert p["seed"] == 99
    assert p["combiner"] == "ridge"


def test_set_params_updates_named_attribute():
    model = SymGeneRegressor(n_genes=3, n_gen=5, verbose=0)
    model.set_params(n_genes=10, n_gen=50)
    assert model.n_genes == 10
    assert model.n_gen == 50


def test_set_params_returns_self():
    model = SymGeneRegressor(verbose=0)
    result = model.set_params(n_genes=4)
    assert result is model


def test_set_params_unknown_key_goes_to_population_kwargs():
    model = SymGeneRegressor(verbose=0)
    model.set_params(cxpb=0.7)
    assert model._population_kwargs.get("cxpb") == 0.7


def test_get_params_reflects_set_params():
    model = SymGeneRegressor(n_genes=3, verbose=0)
    model.set_params(n_genes=12, seed=42)
    p = model.get_params()
    assert p["n_genes"] == 12
    assert p["seed"] == 42


def test_get_params_includes_population_kwargs():
    model = SymGeneRegressor(verbose=0, cxpb=0.8, mutpb=0.1)
    p = model.get_params()
    assert p.get("cxpb") == 0.8
    assert p.get("mutpb") == 0.1


def test_sklearn_clone_compatibility():
    """get_params() deve permitir que sklearn.clone() recrie o estimador."""
    from sklearn.base import clone
    model = SymGeneRegressor(n_genes=4, pop_size=15, n_gen=5, seed=1, verbose=0)
    cloned = clone(model)
    assert cloned.n_genes == model.n_genes
    assert cloned.pop_size == model.pop_size
    assert cloned.seed == model.seed
    assert cloned is not model


def test_sklearn_gridsearchcv_compatibility():
    """SymGeneRegressor deve ser aceito pelo GridSearchCV sem TypeError."""
    from sklearn.model_selection import GridSearchCV
    model = SymGeneRegressor(n_genes=2, pop_size=8, n_gen=2, verbose=0)
    gs = GridSearchCV(model, param_grid={"n_genes": [2, 3]}, cv=2, refit=False)
    gs.fit(X, y)
    assert "param_n_genes" in gs.cv_results_
