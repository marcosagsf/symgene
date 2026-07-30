"""
Ackley 2D — MGGP Surrogate + PSO Demo

Complete surrogate-assisted optimization pipeline:
  1. SymGeneSurrogate samples the Ackley function with 300 LHS points
  2. Fits an MGGP surrogate on the sampled points
  3. PSO minimizes on the surrogate surface
  4. Reports the true function value at the best point found

Demonstrated parameters:
  combiner          — linear combiner of genes: 'ridge', 'lasso' or 'linear'
  ridge_alphas      — list of candidate alphas for Ridge regularization (chosen by CV)
  regression_degree — degree of the regression model over genes (1=linear, 2=quadratic)
  selection         — configurable selection operator (here: TournamentSelection)
  mutpb             — probability of mutation occurring per individual per generation
  mutpb_low         — fraction low-order (subtree) vs high-order (add/remove/replace gene)
  mutation_weights  — weights of [remove, add, replace] in high-order mutation
                      e.g.: [0.5, 1.5, 1.0] → favors adding genes

Run: python -m symgene.examples.ackley_surrogate.run
"""
import numpy as np

from symgene import SymGeneRegressor
from symgene.optimization import PSOOptimizer
from symgene.surrogate import SymGeneSurrogate
from symgene.benchmarks import ackley_2d
from symgene.primitives import STANDARD
from symgene.selection import TournamentSelection
from symgene.metrics import mae, r2, mape


def main():
    bench = ackley_2d()

    regressor = SymGeneRegressor(
        n_genes=4,
        pop_size=60,
        n_gen=80,
        primitives=STANDARD,
        squash={"lim": 8, "alpha": 0.1, "scale": 2.0},
        feature_names=["x1", "x2"],
        seed=0,
        verbose=0,
        # --- combiner and regularization ---
        combiner="ridge",                              # Ridge regularizes gene weights
        ridge_alphas=[0.01, 0.1, 1.0, 10.0, 100.0],  # candidate alphas (best chosen by CV)
        regression_degree=1,                           # linear combination of genes (degree 1)
        # --- selection ---
        selection=TournamentSelection(size=7),         # tournament of size 7
        # --- mutation ---
        mutpb=0.25,                        # 25% chance of mutation per individual per generation
        mutpb_low=0.30,                    # 30% subtree (low), 70% structural (high)
        mutation_weights=[0.5, 1.5, 1.0],  # high-order: remove=0.5, add=1.5, replace=1.0
    )

    optimizer = PSOOptimizer(n_particles=50, n_iter=300, verbose=0)

    surrogate = SymGeneSurrogate(
        regressor=regressor,
        optimizer=optimizer,
        n_samples=300,   # analytic function — sample generously
        maximize=False,
        seed=0,
    )

    result = surrogate.optimize(bench.fn, bounds=bench.bounds)

    # evaluate surrogate quality on independent validation points
    rng = np.random.default_rng(1)
    X_val = rng.uniform(-5, 5, (200, 2))
    y_val  = np.array([bench.fn(X_val[i]) for i in range(200)])
    y_surr = surrogate.predict(X_val)

    print(f"Benchmark      : {bench.name}")
    print(f"Known optimum  : x={bench.x_opt}, f={bench.f_opt}")
    print(f"Surrogate found: x={np.round(result.x_best, 4)}, f_surrogate={result.f_best:.4f}")
    print(f"True f at x    : {result.f_true:.4f}")
    print(f"Gap to optimum : {abs(result.f_true - bench.f_opt):.4f}")
    print(f"Surrogate MAE  : {mae(y_val, y_surr):.4f}")
    print(f"Surrogate R²   : {r2(y_val, y_surr):.4f}")
    print(f"Surrogate MAPE : {mape(y_val, y_surr):.2f}%")
    print(f"Surrogate model: {result.surrogate.best_expression_}")


if __name__ == "__main__":
    main()
