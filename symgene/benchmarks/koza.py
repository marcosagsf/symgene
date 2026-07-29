"""Koza symbolic regression benchmark problems."""
import numpy as np
from typing import NamedTuple


class BenchmarkData(NamedTuple):
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    name: str
    n_inputs: int
    formula: str


def _make_1d(fn, lo: float, hi: float, n_train: int, n_test: int, seed: int):
    rng = np.random.default_rng(seed)
    x = rng.uniform(lo, hi, n_train)
    x_t = np.linspace(lo, hi, n_test)
    return x.reshape(-1, 1), fn(x), x_t.reshape(-1, 1), fn(x_t)


def koza1(n_train: int = 100, n_test: int = 50, seed: int = 0) -> BenchmarkData:
    fn = lambda x: x**4 + x**3 + x**2 + x
    X_tr, y_tr, X_te, y_te = _make_1d(fn, -1.0, 1.0, n_train, n_test, seed)
    return BenchmarkData(X_tr, y_tr, X_te, y_te, "koza1", 1, "x^4 + x^3 + x^2 + x")


def koza2(n_train: int = 100, n_test: int = 50, seed: int = 0) -> BenchmarkData:
    fn = lambda x: x**5 - 2 * x**3 + x
    X_tr, y_tr, X_te, y_te = _make_1d(fn, -1.0, 1.0, n_train, n_test, seed)
    return BenchmarkData(X_tr, y_tr, X_te, y_te, "koza2", 1, "x^5 - 2x^3 + x")


def koza3(n_train: int = 100, n_test: int = 50, seed: int = 0) -> BenchmarkData:
    fn = lambda x: x**6 - 2 * x**4 + x**2
    X_tr, y_tr, X_te, y_te = _make_1d(fn, -1.0, 1.0, n_train, n_test, seed)
    return BenchmarkData(X_tr, y_tr, X_te, y_te, "koza3", 1, "x^6 - 2x^4 + x^2")
