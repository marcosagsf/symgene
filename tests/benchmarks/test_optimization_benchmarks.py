import numpy as np
import pytest
from symgene.benchmarks import forrester_1d, himmelblau_2d, ackley_2d, OptimBenchmark


def test_forrester_is_optimbenchmark():
    b = forrester_1d()
    assert isinstance(b, OptimBenchmark)
    assert b.n_inputs == 1
    assert b.name == "Forrester1D"


def test_forrester_bounds():
    b = forrester_1d()
    assert len(b.bounds) == 1
    assert b.bounds[0] == (0.0, 1.0)


def test_forrester_known_minimum():
    b = forrester_1d()
    f_at_opt = b.fn(b.x_opt)
    assert abs(f_at_opt - b.f_opt) < 0.01


def test_forrester_callable_shape():
    b = forrester_1d()
    val = b.fn(np.array([0.5]))
    assert isinstance(val, float)


def test_himmelblau_is_optimbenchmark():
    b = himmelblau_2d()
    assert isinstance(b, OptimBenchmark)
    assert b.n_inputs == 2


def test_himmelblau_known_minimum():
    b = himmelblau_2d()
    assert abs(b.fn(np.array([3.0, 2.0]))) < 1e-10
    assert abs(b.fn(np.array([-2.805118, 3.131312]))) < 1e-4


def test_himmelblau_bounds():
    b = himmelblau_2d()
    assert len(b.bounds) == 2
    assert all(lo == -5.0 and hi == 5.0 for lo, hi in b.bounds)


def test_ackley_is_optimbenchmark():
    b = ackley_2d()
    assert isinstance(b, OptimBenchmark)
    assert b.n_inputs == 2


def test_ackley_known_minimum():
    b = ackley_2d()
    assert abs(b.fn(np.array([0.0, 0.0]))) < 1e-10


def test_ackley_bounds():
    b = ackley_2d()
    assert len(b.bounds) == 2


def test_forrester_noise_differs():
    b1 = forrester_1d(noise=0.1, seed=0)
    b2 = forrester_1d(noise=0.1, seed=1)
    # Different seeds → different noisy evaluations
    v1 = b1.fn(np.array([0.5]))
    v2 = b2.fn(np.array([0.5]))
    # They may coincidentally match, but over multiple calls they should differ
    vals1 = [b1.fn(np.array([0.5])) for _ in range(5)]
    vals2 = [b2.fn(np.array([0.5])) for _ in range(5)]
    # At least one pair should differ
    assert not all(abs(v1 - v2) < 1e-10 for v1, v2 in zip(vals1, vals2))


def test_all_benchmarks_importable():
    from symgene.benchmarks import forrester_1d, himmelblau_2d, ackley_2d, OptimBenchmark
    assert forrester_1d is not None
    assert himmelblau_2d is not None
    assert ackley_2d is not None
