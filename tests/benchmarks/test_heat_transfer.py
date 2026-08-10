import numpy as np
import pytest
from symgene.benchmarks import dittus_boelter
from symgene.benchmarks.koza import BenchmarkData


def test_dittus_boelter_returns_benchmark_data():
    data = dittus_boelter(n_train=50, n_test=20, seed=0)
    assert isinstance(data, BenchmarkData)


def test_dittus_boelter_shapes():
    data = dittus_boelter(n_train=80, n_test=30, seed=0)
    assert data.X_train.shape == (80, 2)
    assert data.y_train.shape == (80,)
    assert data.X_test.shape == (30, 2)
    assert data.y_test.shape == (30,)


def test_dittus_boelter_feature_names():
    data = dittus_boelter()
    assert data.feature_names == ["Re", "Pr"]


def test_dittus_boelter_n_inputs():
    data = dittus_boelter()
    assert data.n_inputs == 2


def test_dittus_boelter_formula_string():
    data = dittus_boelter()
    assert "Re" in data.formula
    assert "Pr" in data.formula
    assert "0.023" in data.formula


def test_dittus_boelter_re_domain():
    data = dittus_boelter(n_train=200, seed=0)
    Re = data.X_train[:, 0]
    assert np.all(Re >= 1e4 - 1), "Re should be >= 10 000"
    assert np.all(Re <= 1e6 + 1), "Re should be <= 1 000 000"


def test_dittus_boelter_pr_domain():
    data = dittus_boelter(n_train=200, seed=0)
    Pr = data.X_train[:, 1]
    assert np.all(Pr >= 0.6), "Pr should be >= ~0.7"
    assert np.all(Pr <= 200), "Pr should be <= ~160"


def test_dittus_boelter_formula_correctness():
    data = dittus_boelter(n_train=10, seed=42)
    Re = data.X_train[:, 0]
    Pr = data.X_train[:, 1]
    expected = 0.023 * Re ** 0.8 * Pr ** 0.4
    np.testing.assert_allclose(data.y_train, expected, rtol=1e-10)


def test_dittus_boelter_nu_positive():
    data = dittus_boelter(n_train=100, seed=0)
    assert np.all(data.y_train > 0)
    assert np.all(data.y_test > 0)


def test_dittus_boelter_deterministic():
    d1 = dittus_boelter(n_train=50, seed=7)
    d2 = dittus_boelter(n_train=50, seed=7)
    np.testing.assert_array_equal(d1.X_train, d2.X_train)
    np.testing.assert_array_equal(d1.y_train, d2.y_train)


def test_dittus_boelter_train_test_differ():
    data = dittus_boelter(n_train=50, n_test=50, seed=0)
    # Train and test should not be identical (different seeds internally)
    assert not np.array_equal(data.X_train, data.X_test)
