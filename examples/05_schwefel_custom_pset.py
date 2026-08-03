"""
Custom PrimitiveSet + Symbolic Export — Schwefel 2D
Surrogate-assisted optimization with a domain-specific custom primitive set.

Library features showcased:
- ARITHMETIC preset          — minimal preset (add, sub, mul, div only)
- add_custom(..., sympy_fn)  — register custom primitives with symbolic mapping
- to_sympy()                 — tree-traversal conversion to a sympy expression
- to_latex()                 — LaTeX rendering of the symbolic expression
- to_callable()              — standalone Python callable for deployment

Run:
    python examples/05_schwefel_custom_pset.py
"""
import numpy as np
import sympy as sp

from symgene import PrimitiveSet, Population, SymGeneEvolver
from symgene.benchmarks import schwefel_2d
from symgene.fitness import FitnessEvaluator
from symgene.metrics import mae, r2, mape
from symgene.metrics.regression import mse
from symgene.optimization import PSOOptimizer
from symgene.primitives import ARITHMETIC
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
    bench = schwefel_2d()
    rng = np.random.default_rng(0)

    lo, hi = np.array([-500., -500.]), np.array([500., 500.])
    X_all = lhs(rng, lo, hi, 400)
    y_all = np.array([bench.fn(X_all[i]) for i in range(400)])

    n_train, n_val = 280, 60
    X_train, y_train = X_all[:n_train],              y_all[:n_train]
    X_val,   y_val   = X_all[n_train:n_train + n_val], y_all[n_train:n_train + n_val]
    X_test,  y_test  = X_all[n_train + n_val:],      y_all[n_train + n_val:]

    print(f"Train:{X_train.shape[0]}  Val:{X_val.shape[0]}  Test:{X_test.shape[0]}")
    print(f"y range: [{y_all.min():.2f}, {y_all.max():.2f}]")

    # ── Custom PrimitiveSet ──────────────────────────────────────────────────
    pset = PrimitiveSet(n_inputs=2, feature_names=["x1", "x2"])
    pset.set_squash(lim=800, alpha=0.01, scale=2.0)
    pset.add_from_catalog(ARITHMETIC + ["sin", "cos"])
    pset.add_custom(
        fn=lambda x: float(np.sqrt(abs(x))),
        arity=1,
        name="sqrtabs",
        sympy_fn=lambda x: sp.sqrt(sp.Abs(x)),
    )
    pset.add_custom(
        fn=lambda x: float(x * np.sin(np.sqrt(abs(x)))),
        arity=1,
        name="xsinqrt",
        sympy_fn=lambda x: x * sp.sin(sp.sqrt(sp.Abs(x))),
    )

    # ── Fit MGGP surrogate ───────────────────────────────────────────────────
    pop = Population(
        name="schwefel",
        pset=pset,
        n_genes=4, pop_size=60,
        combiner="ridge",
        ridge_alphas=[0.01, 0.1, 1.0, 10.0, 100.0],
        regression_degree=1,
        fitness=FitnessEvaluator(metric=mse),
        selection=TournamentSelection(size=7),
        mutpb=0.25, mutpb_low=0.30,
        mutation_weights=[0.5, 1.5, 1.0],
    )
    evolver = SymGeneEvolver(populations=[pop], n_gen=80, seed=0, verbose=0)
    results = evolver.fit(
        X_train, {"schwefel": y_train},
        X_val=X_val, y_val={"schwefel": y_val},
    )
    regressor = results["schwefel"]
    print(f"Genes     : {regressor.n_genes_}")
    print(f"Expression: {regressor.best_expression_[:80]}...")

    # ── Symbolic export ──────────────────────────────────────────────────────
    print("\n--- Symbolic Export ---")
    latex_expr = regressor.to_latex()
    print(f"LaTeX (first 200 chars):\n  {latex_expr[:200]}")

    fn_callable = regressor.to_callable()
    x_opt = np.array([[420.9687, 420.9687]])
    print(f"\nto_callable() at optimum (420.97, 420.97):")
    print(f"  Surrogate prediction : {fn_callable(x_opt)[0]:.4f}")
    print(f"  True f at optimum    : {bench.fn(x_opt[0]):.6f}")

    # ── PSO on surrogate ─────────────────────────────────────────────────────
    optimizer = PSOOptimizer(n_particles=50, n_iter=300, verbose=0)
    pso_result = optimizer.optimize(
        lambda x: float(regressor.predict(x.reshape(1, -1))[0]),
        bounds=bench.bounds, seed=0,
    )
    f_true = float(bench.fn(pso_result.x_best))

    # ── Results ──────────────────────────────────────────────────────────────
    yp_test = regressor.predict(X_test)
    print(f"\nSurrogate R²   (test) : {r2(y_test, yp_test):.4f}")
    print(f"Surrogate MAE  (test) : {mae(y_test, yp_test):.4f}")
    print(f"Surrogate MAPE (test) : {mape(y_test, yp_test):.2f}%")
    print(f"\nPSO found x       : {np.round(pso_result.x_best, 6)}")
    print(f"Surrogate f(x*)   : {pso_result.f_best:.6f}")
    print(f"True f(x*)        : {f_true:.6f}")
    print(f"Known optimum     : f={bench.f_opt}  at x={bench.x_opt}")
    print(f"Gap to optimum    : {abs(f_true - bench.f_opt):.6f}")


if __name__ == "__main__":
    main()
