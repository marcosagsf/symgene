"""
Dittus-Boelter correlation for convective heat transfer.

Nu = 0.023 · Re^0.8 · Pr^0.4

This is the standard engineering correlation for the Nusselt number (Nu)
in turbulent pipe flow, relating convective heat transfer to the Reynolds
number (Re, flow regime) and Prandtl number (Pr, fluid property).

Domain:
  Re ∈ [10 000, 1 000 000]  — fully turbulent regime
  Pr ∈ [0.7,       160]     — air (≈0.7) to engine oil (≈160)

Sampling is log-uniform via Latin Hypercube for both variables so the
orders-of-magnitude span of Re is represented evenly.
"""
import numpy as np
from symgene.benchmarks.koza import BenchmarkData


def _lhs_log(rng: np.random.Generator, log_lo: float, log_hi: float, n: int) -> np.ndarray:
    """1-D log-uniform Latin Hypercube sample of size n."""
    perm = rng.permutation(n)
    u = (perm + rng.uniform(0.0, 1.0, n)) / n
    return 10.0 ** (log_lo + u * (log_hi - log_lo))


def _sample(rng: np.random.Generator, n: int) -> np.ndarray:
    Re = _lhs_log(rng, 4.0, 6.0, n)        # 1e4 → 1e6
    Pr = _lhs_log(rng, -0.155, 2.204, n)   # 0.7 → 160
    return np.column_stack([Re, Pr])


def _nu(X: np.ndarray) -> np.ndarray:
    return 0.023 * X[:, 0] ** 0.8 * X[:, 1] ** 0.4


def dittus_boelter(
    n_train: int = 200,
    n_test: int = 100,
    seed: int = 0,
) -> BenchmarkData:
    """Dittus-Boelter heat transfer correlation benchmark.

    Generates training and test sets for the symbolic regression task of
    recovering the Nusselt-number correlation from (Re, Pr) observations.

    Parameters
    ----------
    n_train : int
        Number of training samples.
    n_test : int
        Number of test samples.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    BenchmarkData
        Named tuple with X_train, y_train, X_test, y_test, name, n_inputs,
        formula, and feature_names=["Re", "Pr"].

    Notes
    -----
    The target expression involves power-law exponents (0.8 and 0.4) that are
    not integers. A well-configured PrimitiveSet should include the ``pow``
    primitive and ephemeral constants covering [0.0, 1.5] to allow the GP to
    discover the correct exponents.

    Examples
    --------
    >>> from symgene.benchmarks import dittus_boelter
    >>> data = dittus_boelter(n_train=200, n_test=100)
    >>> data.feature_names
    ['Re', 'Pr']
    >>> data.formula
    '0.023 * Re^0.8 * Pr^0.4'
    """
    rng_tr = np.random.default_rng(seed)
    rng_te = np.random.default_rng(seed + 1_000_003)   # uncorrelated split

    X_tr = _sample(rng_tr, n_train)
    X_te = _sample(rng_te, n_test)

    return BenchmarkData(
        X_train=X_tr,
        y_train=_nu(X_tr),
        X_test=X_te,
        y_test=_nu(X_te),
        name="dittus_boelter",
        n_inputs=2,
        formula="0.023 * Re^0.8 * Pr^0.4",
        feature_names=["Re", "Pr"],
    )
