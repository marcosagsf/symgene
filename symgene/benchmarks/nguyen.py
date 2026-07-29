"""Nguyen symbolic regression benchmark suite (nguyen1–nguyen10)."""
import numpy as np
from symgene.benchmarks.koza import BenchmarkData, _make_1d


def nguyen1(n_train: int = 100, n_test: int = 50, seed: int = 0) -> BenchmarkData:
    fn = lambda x: x**3 + x**2 + x
    X_tr, y_tr, X_te, y_te = _make_1d(fn, -1.0, 1.0, n_train, n_test, seed)
    return BenchmarkData(X_tr, y_tr, X_te, y_te, "nguyen1", 1, "x^3 + x^2 + x")


def nguyen2(n_train: int = 100, n_test: int = 50, seed: int = 0) -> BenchmarkData:
    fn = lambda x: x**4 + x**3 + x**2 + x
    X_tr, y_tr, X_te, y_te = _make_1d(fn, -1.0, 1.0, n_train, n_test, seed)
    return BenchmarkData(X_tr, y_tr, X_te, y_te, "nguyen2", 1, "x^4 + x^3 + x^2 + x")


def nguyen3(n_train: int = 100, n_test: int = 50, seed: int = 0) -> BenchmarkData:
    fn = lambda x: x**5 + x**4 + x**3 + x**2 + x
    X_tr, y_tr, X_te, y_te = _make_1d(fn, -1.0, 1.0, n_train, n_test, seed)
    return BenchmarkData(X_tr, y_tr, X_te, y_te, "nguyen3", 1, "x^5 + x^4 + x^3 + x^2 + x")


def nguyen4(n_train: int = 100, n_test: int = 50, seed: int = 0) -> BenchmarkData:
    fn = lambda x: x**6 + x**5 + x**4 + x**3 + x**2 + x
    X_tr, y_tr, X_te, y_te = _make_1d(fn, -1.0, 1.0, n_train, n_test, seed)
    return BenchmarkData(X_tr, y_tr, X_te, y_te, "nguyen4", 1, "x^6 + x^5 + x^4 + x^3 + x^2 + x")


def nguyen5(n_train: int = 100, n_test: int = 50, seed: int = 0) -> BenchmarkData:
    fn = lambda x: np.sin(x**2) * np.cos(x) - 1
    X_tr, y_tr, X_te, y_te = _make_1d(fn, -1.0, 1.0, n_train, n_test, seed)
    return BenchmarkData(X_tr, y_tr, X_te, y_te, "nguyen5", 1, "sin(x^2)cos(x) - 1")


def nguyen6(n_train: int = 100, n_test: int = 50, seed: int = 0) -> BenchmarkData:
    fn = lambda x: np.sin(x) + np.sin(x + x**2)
    X_tr, y_tr, X_te, y_te = _make_1d(fn, -1.0, 1.0, n_train, n_test, seed)
    return BenchmarkData(X_tr, y_tr, X_te, y_te, "nguyen6", 1, "sin(x) + sin(x + x^2)")


def nguyen7(n_train: int = 100, n_test: int = 50, seed: int = 0) -> BenchmarkData:
    fn = lambda x: np.log(x + 1) + np.log(x**2 + 1)
    X_tr, y_tr, X_te, y_te = _make_1d(fn, 0.01, 2.0, n_train, n_test, seed)
    return BenchmarkData(X_tr, y_tr, X_te, y_te, "nguyen7", 1, "log(x+1) + log(x^2+1)")


def nguyen8(n_train: int = 100, n_test: int = 50, seed: int = 0) -> BenchmarkData:
    fn = lambda x: np.sqrt(x)
    X_tr, y_tr, X_te, y_te = _make_1d(fn, 0.0, 4.0, n_train, n_test, seed)
    return BenchmarkData(X_tr, y_tr, X_te, y_te, "nguyen8", 1, "sqrt(x)")


def nguyen9(n_train: int = 100, n_test: int = 50, seed: int = 0) -> BenchmarkData:
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-1.0, 1.0, (n_train, 2))
    y_tr = np.sin(xy[:, 0]) + np.sin(xy[:, 1] ** 2)
    xy_t = rng.uniform(-1.0, 1.0, (n_test, 2))
    y_te = np.sin(xy_t[:, 0]) + np.sin(xy_t[:, 1] ** 2)
    return BenchmarkData(xy, y_tr, xy_t, y_te, "nguyen9", 2, "sin(x) + sin(y^2)")


def nguyen10(n_train: int = 100, n_test: int = 50, seed: int = 0) -> BenchmarkData:
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-1.0, 1.0, (n_train, 2))
    y_tr = 2 * np.sin(xy[:, 0]) * np.cos(xy[:, 1])
    xy_t = rng.uniform(-1.0, 1.0, (n_test, 2))
    y_te = 2 * np.sin(xy_t[:, 0]) * np.cos(xy_t[:, 1])
    return BenchmarkData(xy, y_tr, xy_t, y_te, "nguyen10", 2, "2sin(x)cos(y)")
