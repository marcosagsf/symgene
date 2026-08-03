"""
Surrogate MGGP + PSO — Himmelblau 2D
Surrogate-assisted optimization: MGGP fits a surrogate on 300 LHS samples,
then PSO minimizes on the surrogate surface.

Library features showcased:
- combiner="lasso"        — Lasso regression combiner
- RouletteSelection       — fitness-proportionate roulette selection
- regression_degree=2     — degree-2 polynomial gene features
- n_genes_max             — upper bound on gene count during evolution
- elite_ratio             — fraction of elite individuals preserved each generation

Run:
    python examples/02_surrogate_himmelblau.py
"""
import numpy as np

from symgene import SymGeneRegressor
from symgene.benchmarks import himmelblau_2d
from symgene.metrics import mae, r2, mape
from symgene.optimization import PSOOptimizer
from symgene.primitives import STANDARD
from symgene.selection.roulette import RouletteSelection


def lhs(rng, lo, hi, n):
    d = len(lo)
    X = np.empty((n, d))
    for j in range(d):
        perm = rng.permutation(n)
        X[:, j] = lo[j] + (perm + rng.uniform(0, 1, n)) / n * (hi[j] - lo[j])
    return X


def main():
    # ── Data generation (70 / 15 / 15 split from 300 LHS samples) ───────────
    bench = himmelblau_2d()
    rng = np.random.default_rng(0)

    lo, hi = np.array([-5., -5.]), np.array([5., 5.])
    X_all = lhs(rng, lo, hi, 300)
    y_all = np.array([bench.fn(X_all[i]) for i in range(300)])

    n_train, n_val = 210, 45
    X_train, y_train = X_all[:n_train],              y_all[:n_train]
    X_val,   y_val   = X_all[n_train:n_train + n_val], y_all[n_train:n_train + n_val]
    X_test,  y_test  = X_all[n_train + n_val:],      y_all[n_train + n_val:]

    print(f"Train:{X_train.shape[0]}  Val:{X_val.shape[0]}  Test:{X_test.shape[0]}")
    print(f"y range: [{y_all.min():.2f}, {y_all.max():.2f}]")

    # ── Fit MGGP surrogate ───────────────────────────────────────────────────
    regressor = SymGeneRegressor(
        n_genes=3, pop_size=80, n_gen=100,
        primitives=STANDARD,
        squash={"lim": 8, "alpha": 0.08, "scale": 2.0},
        feature_names=["x1", "x2"],
        seed=42, verbose=0,
        combiner="lasso",
        regression_degree=2,
        selection=RouletteSelection(),
        n_genes_max=8,
        elite_ratio=0.04,
        mutpb=0.28, mutpb_low=0.18,
        mutation_weights=[0.4, 1.5, 0.8],
        tree_max=30, height_max=6,
    )
    regressor.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    print(f"Genes     : {regressor.n_genes_}")
    print(f"Expression: {regressor.best_expression_}")

    # ── PSO on surrogate ─────────────────────────────────────────────────────
    optimizer = PSOOptimizer(n_particles=80, n_iter=300, verbose=0)
    pso_result = optimizer.optimize(
        lambda x: float(regressor.predict(x.reshape(1, -1))[0]),
        bounds=bench.bounds, seed=0,
    )
    f_true = float(bench.fn(pso_result.x_best))

    known_optima = [
        ( 3.000000,  2.000000),
        (-2.805118,  3.131312),
        (-3.779310, -3.283186),
        ( 3.584428, -1.848126),
    ]

    # ── Results ──────────────────────────────────────────────────────────────
    yp_test = regressor.predict(X_test)
    print(f"\nSurrogate R²   (test) : {r2(y_test, yp_test):.4f}")
    print(f"Surrogate MAE  (test) : {mae(y_test, yp_test):.4f}")
    print(f"Surrogate MAPE (test) : {mape(y_test, yp_test):.2f}%")
    print(f"\nPSO found x       : {np.round(pso_result.x_best, 4)}")
    print(f"Surrogate f(x*)   : {pso_result.f_best:.6f}")
    print(f"True f(x*)        : {f_true:.6f}")
    print(f"Known optima      : f=0 at 4 locations (Himmelblau)")
    dist = min(
        np.hypot(pso_result.x_best[0] - ox, pso_result.x_best[1] - oy)
        for ox, oy in known_optima
    )
    print(f"Dist. to nearest  : {dist:.6f}")


if __name__ == "__main__":
    main()
