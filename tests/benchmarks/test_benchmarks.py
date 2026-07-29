import numpy as np
import pytest
from symgene.benchmarks import (
    koza1, koza2, koza3,
    nguyen1, nguyen2, nguyen3, nguyen4, nguyen5,
    nguyen6, nguyen7, nguyen8, nguyen9, nguyen10,
)

ALL_BENCHMARKS = [
    koza1, koza2, koza3,
    nguyen1, nguyen2, nguyen3, nguyen4, nguyen5,
    nguyen6, nguyen7, nguyen8, nguyen9, nguyen10,
]


def test_all_benchmarks_return_four_arrays():
    for fn in ALL_BENCHMARKS:
        data = fn(n_train=10, n_test=5, seed=0)
        assert data.X_train.shape[0] == 10
        assert data.y_train.shape[0] == 10
        assert data.X_test.shape[0] == 5
        assert data.y_test.shape[0] == 5


def test_koza1_formula():
    data = koza1(n_train=5, seed=42)
    x = data.X_train.ravel()
    np.testing.assert_allclose(data.y_train, x**4 + x**3 + x**2 + x)


def test_koza2_formula():
    data = koza2(n_train=5, seed=42)
    x = data.X_train.ravel()
    np.testing.assert_allclose(data.y_train, x**5 - 2 * x**3 + x)


def test_koza3_formula():
    data = koza3(n_train=5, seed=0)
    x = data.X_train.ravel()
    np.testing.assert_allclose(data.y_train, x**6 - 2 * x**4 + x**2)


def test_nguyen1_formula():
    data = nguyen1(n_train=5, seed=42)
    x = data.X_train.ravel()
    np.testing.assert_allclose(data.y_train, x**3 + x**2 + x)


def test_nguyen5_formula():
    data = nguyen5(n_train=5, seed=42)
    x = data.X_train.ravel()
    np.testing.assert_allclose(data.y_train, np.sin(x**2) * np.cos(x) - 1)


def test_nguyen7_domain_positive():
    data = nguyen7(n_train=20, seed=0)
    assert np.all(data.X_train > 0)


def test_nguyen8_domain_nonneg():
    data = nguyen8(n_train=20, seed=0)
    assert np.all(data.X_train >= 0)


def test_nguyen9_two_inputs():
    data = nguyen9(n_train=20, n_test=10)
    assert data.X_train.shape == (20, 2)
    assert data.n_inputs == 2


def test_nguyen10_two_inputs():
    data = nguyen10(n_train=20, n_test=10)
    assert data.X_train.shape == (20, 2)
    assert data.n_inputs == 2


def test_koza_n_inputs():
    for fn in [koza1, koza2, koza3]:
        data = fn()
        assert data.n_inputs == 1


def test_benchmark_named_tuple_fields():
    data = koza1()
    assert hasattr(data, "name")
    assert hasattr(data, "formula")
    assert isinstance(data.formula, str)


def test_benchmark_determinism():
    d1 = koza1(n_train=20, seed=7)
    d2 = koza1(n_train=20, seed=7)
    np.testing.assert_array_equal(d1.X_train, d2.X_train)
