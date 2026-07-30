import numpy as np
import pytest
from symgene.optimization import PSOOptimizer, OptimResult


def sphere(x):
    return float(np.sum(x ** 2))


def himmelblau(x):
    return float((x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2)


def test_optimresult_fields():
    result = OptimResult(
        x_best=np.array([1.0, 2.0]),
        f_best=0.5,
        history=[1.0, 0.5],
        n_eval=60,
    )
    assert result.f_best == 0.5
    assert len(result.history) == 2
    assert result.n_eval == 60


def test_pso_returns_optimresult():
    pso = PSOOptimizer(n_particles=10, n_iter=5)
    result = pso.optimize(sphere, bounds=[(-5, 5), (-5, 5)], seed=0)
    assert isinstance(result, OptimResult)
    assert result.x_best.shape == (2,)
    assert isinstance(result.f_best, float)


def test_pso_minimizes_sphere():
    pso = PSOOptimizer(n_particles=20, n_iter=50, seed=0)
    result = pso.optimize(sphere, bounds=[(-5, 5), (-5, 5)], seed=0)
    assert result.f_best < 1.0  # should get close to 0


def test_pso_history_length():
    n_iter = 20
    pso = PSOOptimizer(n_particles=10, n_iter=n_iter)
    result = pso.optimize(sphere, bounds=[(-2, 2)], seed=0)
    assert len(result.history) == n_iter + 1  # initial + one per iter


def test_pso_history_non_increasing():
    pso = PSOOptimizer(n_particles=10, n_iter=30)
    result = pso.optimize(sphere, bounds=[(-5, 5), (-5, 5)], seed=42)
    for i in range(1, len(result.history)):
        assert result.history[i] <= result.history[i - 1] + 1e-12


def test_pso_n_eval():
    n_particles, n_iter = 10, 5
    pso = PSOOptimizer(n_particles=n_particles, n_iter=n_iter)
    result = pso.optimize(sphere, bounds=[(-2, 2), (-2, 2)], seed=0)
    assert result.n_eval == n_particles * (n_iter + 1)


def test_pso_respects_bounds():
    pso = PSOOptimizer(n_particles=15, n_iter=20)
    result = pso.optimize(sphere, bounds=[(1.0, 2.0), (1.0, 2.0)], seed=0)
    # best x must be within bounds
    assert np.all(result.x_best >= 1.0)
    assert np.all(result.x_best <= 2.0)


def test_pso_seed_reproducible():
    pso = PSOOptimizer(n_particles=10, n_iter=20)
    r1 = pso.optimize(sphere, bounds=[(-5, 5), (-5, 5)], seed=7)
    r2 = pso.optimize(sphere, bounds=[(-5, 5), (-5, 5)], seed=7)
    assert r1.f_best == r2.f_best
    np.testing.assert_array_equal(r1.x_best, r2.x_best)


def test_pso_himmelblau():
    # Himmelblau has 4 global minima at f=0
    pso = PSOOptimizer(n_particles=30, n_iter=100)
    result = pso.optimize(himmelblau, bounds=[(-5, 5), (-5, 5)], seed=0)
    assert result.f_best < 0.1


def test_pso_1d():
    # 1D: f(x) = (x-3)^2, minimum at x=3
    fn = lambda x: float((x[0] - 3.0) ** 2)
    pso = PSOOptimizer(n_particles=20, n_iter=50)
    result = pso.optimize(fn, bounds=[(0, 6)], seed=0)
    assert abs(result.x_best[0] - 3.0) < 0.1
