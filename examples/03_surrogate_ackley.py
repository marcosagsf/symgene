"""
Surrogate MGGP + PSO — Ackley 2D
Surrogate-assisted optimization: MGGP fits a surrogate on 400 LHS samples,
then PSO minimizes on the surrogate surface.

Library features showcased:
- combiner="ridge"         — Ridge regression combiner
- TournamentSelection      — tournament selection
- regression_degree=1      — linear combination of gene outputs

Run:
    python examples/03_surrogate_ackley.py
"""
import numpy as np

from symgene import SymGeneRegressor
from symgene.benchmarks import ackley_2d
from symgene.metrics import mae, r2, mape
from symgene.optimization import PSOOptimizer
from symgene.primitives import STANDARD
from symgene.selection import TournamentSelection


def lhs(rng, lo, hi, n):
    d = len(lo)
    X = np.empty((n, d))
    for j in range(d):
        perm = rng.permutation(n)
        X[:, j] = lo[j] + (perm + rng.uniform(0, 1, n)) / n * (hi[j] - lo[j])
    return X


def main():
    # ── Data generation (70 / 15 / 15 split from 400 LHS samples) ───────────
    bench = ackley_2d()
    rng = np.random.default_rng(0)

    lo, hi = np.array([-5., -5.]), np.array([5., 5.])
    X_all = lhs(rng, lo, hi, 400)
    y_all = np.array([bench.fn(X_all[i]) for i in range(400)])

    n_train, n_val = 280, 60
    X_train, y_train = X_all[:n_train],              y_all[:n_train]
    X_val,   y_val   = X_all[n_train:n_train + n_val], y_all[n_train:n_train + n_val]
    X_test,  y_test  = X_all[n_train + n_val:],      y_all[n_train + n_val:]

    print(f"Train:{X_train.shape[0]}  Val:{X_val.shape[0]}  Test:{X_test.shape[0]}")
    print(f"y range: [{y_all.min():.3f}, {y_all.max():.3f}]")

    # ── Fit MGGP surrogate ───────────────────────────────────────────────────
    reg = SymGeneRegressor(
        n_genes=4, pop_size=80, n_gen=200,
        primitives=STANDARD,
        squash={"lim": 8, "alpha": 0.1, "scale": 2.0},
        feature_names=["x1", "x2"],
        seed=0, verbose=0,
        combiner="ridge",
        ridge_alphas=[0.01, 0.1, 1.0, 10.0, 100.0],
        regression_degree=1,
        selection=TournamentSelection(size=4),
        mutpb=0.25, mutpb_low=0.30,
        mutation_weights=[0.5, 1.5, 1.0],
    )
    reg.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    print(f"Genes     : {reg.n_genes_}")
    print(f"Expression: {reg.best_expression_[:120]}...")

    # ── PSO on surrogate ─────────────────────────────────────────────────────
    optimizer = PSOOptimizer(n_particles=50, n_iter=300, verbose=0)
    pso_res = optimizer.optimize(
        lambda x: float(reg.predict(x.reshape(1, -1))[0]),
        bounds=bench.bounds, seed=0,
    )
    f_true = float(bench.fn(pso_res.x_best))

    # ── Results ──────────────────────────────────────────────────────────────
    yp_te = reg.predict(X_test)
    print(f"\nSurrogate R²   (test) : {r2(y_test,   yp_te):.4f}")
    print(f"Surrogate MAE  (test) : {mae(y_test,  yp_te):.4f}")
    print(f"Surrogate MAPE (test) : {mape(y_test, yp_te):.2f}%")
    print(f"\nPSO found x       : {np.round(pso_res.x_best, 6)}")
    print(f"Surrogate f(x*)   : {pso_res.f_best:.6f}")
    print(f"True f(x*)        : {f_true:.6f}")
    print(f"True optimum      : f={bench.f_opt}  at x={bench.x_opt}")
    print(f"Gap to optimum    : {abs(f_true - bench.f_opt):.6f}")


if __name__ == "__main__":
    main()
