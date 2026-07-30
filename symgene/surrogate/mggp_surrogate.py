"""
Surrogate-assisted optimization using MGGP as the surrogate model.

Workflow:
  1. Sample the objective function at training points
  2. Fit a SymGeneRegressor surrogate on those samples
  3. Use an Optimizer to minimize/maximize on the surrogate
  4. Return the best point found, plus the surrogate model
"""
from __future__ import annotations
from typing import Callable, Sequence
from dataclasses import dataclass, field
import numpy as np

from symgene.regressor import SymGeneRegressor
from symgene.optimization.base import Optimizer, OptimResult


@dataclass
class SurrogateResult:
    x_best: np.ndarray        # best solution found on surrogate
    f_best: float             # surrogate value at x_best
    f_true: float | None      # true objective value at x_best (if fn provided to optimize)
    optim_result: OptimResult # full optimization trace on surrogate
    surrogate: SymGeneRegressor  # fitted surrogate model


class SymGeneSurrogate:
    """
    MGGP surrogate-assisted optimizer.

    Parameters
    ----------
    regressor : SymGeneRegressor
        MGGP model used as surrogate. Caller configures it fully.
    optimizer : Optimizer
        Metaheuristic optimizer (e.g. PSOOptimizer) used to search the surrogate.
    n_samples : int
        Number of random Latin-hypercube samples used to train the surrogate.
    maximize : bool
        If True, maximizes the objective (internally negates for optimizer).
    seed : int | None
        Random seed for sampling.
    """

    def __init__(
        self,
        regressor: SymGeneRegressor,
        optimizer: Optimizer,
        n_samples: int = 50,
        maximize: bool = False,
        seed: int | None = None,
    ):
        self.regressor = regressor
        self.optimizer = optimizer
        self.n_samples = n_samples
        self.maximize = maximize
        self.seed = seed

        self.surrogate_: SymGeneRegressor | None = None
        self.X_train_: np.ndarray | None = None
        self.y_train_: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SymGeneSurrogate":
        """Fit the surrogate on provided (X, y) data."""
        self.X_train_ = X
        self.y_train_ = y
        self.surrogate_ = self.regressor
        self.surrogate_.fit(X, y)
        return self

    def optimize(
        self,
        fn: Callable[[np.ndarray], float],
        bounds: Sequence[tuple[float, float]],
    ) -> SurrogateResult:
        """
        Full surrogate-assisted optimization pipeline:
        1. Sample fn at n_samples points within bounds (LHS).
        2. Fit surrogate on samples.
        3. Optimize surrogate with self.optimizer.
        4. Evaluate true fn at best point found.

        Parameters
        ----------
        fn : callable
            True objective function fn(x) -> float (to minimize unless maximize=True).
        bounds : sequence of (lo, hi) tuples
            Search domain, one per dimension.
        """
        rng = np.random.default_rng(self.seed)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        # --- Step 1: Latin-hypercube sampling ---
        X_train = self._lhs(rng, lo, hi, self.n_samples)
        sign = -1.0 if self.maximize else 1.0
        y_train = np.array([sign * fn(X_train[i]) for i in range(self.n_samples)])

        # --- Step 2: Fit surrogate ---
        self.X_train_ = X_train
        self.y_train_ = y_train
        self.surrogate_ = self.regressor
        self.surrogate_.fit(X_train, y_train)

        # --- Step 3: Optimize on surrogate ---
        def surrogate_fn(x: np.ndarray) -> float:
            return float(self.surrogate_.predict(x.reshape(1, -1))[0])

        optim_result = self.optimizer.optimize(surrogate_fn, bounds, seed=self.seed)

        # --- Step 4: Evaluate true function at best point ---
        f_true = sign * fn(optim_result.x_best)

        return SurrogateResult(
            x_best=optim_result.x_best,
            f_best=optim_result.f_best,
            f_true=float(f_true),
            optim_result=optim_result,
            surrogate=self.surrogate_,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the fitted surrogate."""
        if self.surrogate_ is None:
            raise RuntimeError("Call fit() or optimize() first.")
        return self.surrogate_.predict(X)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _lhs(rng: np.random.Generator, lo: np.ndarray, hi: np.ndarray, n: int) -> np.ndarray:
        """Simple Latin-hypercube sample in [lo, hi]."""
        d = len(lo)
        result = np.empty((n, d))
        for j in range(d):
            perm = rng.permutation(n)
            result[:, j] = lo[j] + (perm + rng.uniform(0, 1, n)) / n * (hi[j] - lo[j])
        return result
