"""
Ackley 2D — MGGP Surrogate + PSO Demo
Full surrogate-assisted optimization pipeline:
  1. SymGeneSurrogate samples the Ackley function at 60 LHS points
  2. Fits a MGGP surrogate
  3. PSO minimizes on the surrogate
  4. Reports true function value at best point found
Run: python -m symgene.examples.ackley_surrogate.run
"""
import numpy as np
from symgene import SymGeneRegressor
from symgene.optimization import PSOOptimizer
from symgene.surrogate import SymGeneSurrogate
from symgene.benchmarks import ackley_2d
from symgene.primitives import STANDARD


def main():
    bench = ackley_2d()

    regressor = SymGeneRegressor(
        n_genes=4,
        pop_size=40,
        n_gen=30,
        primitives=STANDARD,
        squash={"lim": 8, "alpha": 0.1, "scale": 2.0},
        feature_names=["x1", "x2"],
        seed=0,
        verbose=0,
    )
    optimizer = PSOOptimizer(n_particles=40, n_iter=150, verbose=0)

    surrogate = SymGeneSurrogate(
        regressor=regressor,
        optimizer=optimizer,
        n_samples=60,
        maximize=False,
        seed=0,
    )

    result = surrogate.optimize(bench.fn, bounds=bench.bounds)

    print(f"Benchmark      : {bench.name}")
    print(f"Known optimum  : x={bench.x_opt}, f={bench.f_opt}")
    print(f"Surrogate found: x={np.round(result.x_best, 4)}, f_surrogate={result.f_best:.4f}")
    print(f"True f at x    : {result.f_true:.4f}")
    print(f"Surrogate model: {result.surrogate.best_expression_}")
    print(f"Gap to optimum : {abs(result.f_true - bench.f_opt):.4f}")


if __name__ == "__main__":
    main()
