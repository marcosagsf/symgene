import numpy as np
import pytest
from symgene.surrogate import SymGeneSurrogate
from symgene.optimization import PSOOptimizer
from symgene import SymGeneRegressor
from symgene.primitives import STANDARD


def make_regressor():
    return SymGeneRegressor(
        n_genes=3,
        pop_size=20,
        n_gen=10,
        primitives=STANDARD,
        seed=0,
        verbose=0,
    )


def make_pso():
    return PSOOptimizer(n_particles=20, n_iter=30)


def sphere(x):
    return float(np.sum(x ** 2))


def forrester(x):
    xv = float(x[0])
    return float((6 * xv - 2) ** 2 * np.sin(12 * xv - 4))


def test_surrogate_result_fields():
    from symgene.surrogate.mggp_surrogate import SurrogateResult
    import numpy as np
    r = SurrogateResult(
        x_best=np.array([1.0]),
        f_best=0.5,
        f_true=0.6,
        optim_result=None,
        surrogate=None,
    )
    assert r.f_best == 0.5
    assert r.f_true == 0.6


def test_surrogate_fit_stores_data():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (30, 1))
    y = np.array([forrester(X[i]) for i in range(30)])
    s = SymGeneSurrogate(make_regressor(), make_pso(), seed=0)
    s.fit(X, y)
    assert s.X_train_ is not None
    assert s.surrogate_ is not None


def test_surrogate_predict_after_fit():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (30, 1))
    y = np.array([forrester(X[i]) for i in range(30)])
    s = SymGeneSurrogate(make_regressor(), make_pso(), seed=0)
    s.fit(X, y)
    preds = s.predict(X)
    assert preds.shape == (30,)


def test_surrogate_predict_before_fit_raises():
    s = SymGeneSurrogate(make_regressor(), make_pso())
    with pytest.raises(RuntimeError):
        s.predict(np.array([[0.5]]))


def test_surrogate_optimize_returns_result():
    from symgene.surrogate.mggp_surrogate import SurrogateResult
    s = SymGeneSurrogate(make_regressor(), make_pso(), n_samples=30, seed=0)
    result = s.optimize(forrester, bounds=[(0.0, 1.0)])
    assert isinstance(result, SurrogateResult)
    assert result.x_best.shape == (1,)
    assert isinstance(result.f_true, float)


def test_surrogate_optimize_bounds_respected():
    s = SymGeneSurrogate(make_regressor(), make_pso(), n_samples=20, seed=0)
    result = s.optimize(sphere, bounds=[(1.0, 2.0), (1.0, 2.0)])
    assert np.all(result.x_best >= 1.0)
    assert np.all(result.x_best <= 2.0)


def test_surrogate_lhs_coverage():
    from symgene.surrogate.mggp_surrogate import SymGeneSurrogate as SS
    rng = np.random.default_rng(0)
    lo = np.array([0.0, 0.0])
    hi = np.array([1.0, 1.0])
    X = SS._lhs(rng, lo, hi, n=20)
    assert X.shape == (20, 2)
    assert np.all(X >= 0.0) and np.all(X <= 1.0)


def test_surrogate_maximize():
    from symgene.surrogate.mggp_surrogate import SurrogateResult
    # Negate sphere: maximum at (0,0)
    neg_sphere = lambda x: -float(np.sum(x ** 2))
    s = SymGeneSurrogate(make_regressor(), make_pso(), n_samples=20, maximize=True, seed=0)
    result = s.optimize(neg_sphere, bounds=[(-3.0, 3.0), (-3.0, 3.0)])
    assert isinstance(result, SurrogateResult)
