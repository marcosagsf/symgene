"""
Himmelblau 2D — PSO Standalone Demo
Uses PSOOptimizer directly (no surrogate) to find one of the four
global minima of the Himmelblau function.
Run: python -m symgene.examples.himmelblau_pso.run
"""
import numpy as np
from symgene.optimization import PSOOptimizer
from symgene.benchmarks import himmelblau_2d


def main():
    bench = himmelblau_2d()

    pso = PSOOptimizer(n_particles=40, n_iter=200, w=0.7, c1=1.5, c2=1.5, verbose=0)
    result = pso.optimize(bench.fn, bounds=bench.bounds, seed=0)

    print(f"Benchmark  : {bench.name}")
    print(f"Formula    : {bench.formula}")
    print(f"Known optima: f=0.0 at (3,2), (-2.805,3.131), (-3.779,-3.283), (3.584,-1.848)")
    print(f"PSO found  : x={result.x_best}, f={result.f_best:.6f}")
    print(f"Evaluations: {result.n_eval}")
    print(f"Converged  : {'YES' if result.f_best < 1e-3 else 'CLOSE'}")


if __name__ == "__main__":
    main()
