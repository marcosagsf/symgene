"""
Ackley 2D — MGGP Surrogate + PSO Demo
Pipeline completo de otimização assistida por surrogate:
  1. SymGeneSurrogate amostra a função Ackley com 300 pontos LHS
  2. Ajusta um surrogate MGGP sobre os pontos amostrados
  3. PSO minimiza na superfície do surrogate
  4. Reporta o valor real da função no melhor ponto encontrado

Parâmetros demonstrados:
  mutpb            — probabilidade de mutação ocorrer por geração
  mutpb_low        — fração da mutação que é low-order (subtree)
                     vs high-order (add/remove/replace de gene)
  mutation_weights — pesos relativos de [remove, add, replace] na mutação high-order
                     ex: [0.5, 1.5, 1.0] → favorece adicionar genes, reduz remoção

Run: python -m symgene.examples.ackley_surrogate.run
"""
import numpy as np
from symgene import SymGeneRegressor
from symgene.optimization import PSOOptimizer
from symgene.surrogate import SymGeneSurrogate
from symgene.benchmarks import ackley_2d
from symgene.primitives import STANDARD
from symgene.metrics import mae


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
        # --- controle de mutação ---
        mutpb=0.25,                        # 25% de chance de mutação por indivíduo por geração
        mutpb_low=0.30,                    # 30% low-order (subtree), 70% high-order (estrutural)
        mutation_weights=[0.5, 1.5, 1.0],  # high-order: remove(0.5) add(1.5) replace(1.0)
                                           # favorece adicionar genes → modelos mais expressivos
    )
    optimizer = PSOOptimizer(n_particles=50, n_iter=300, verbose=0)

    surrogate = SymGeneSurrogate(
        regressor=regressor,
        optimizer=optimizer,
        n_samples=300,   # função analítica — amostrar generosamente
        maximize=False,
        seed=0,
    )

    result = surrogate.optimize(bench.fn, bounds=bench.bounds)

    # Avalia qualidade do surrogate sobre grid de teste
    rng = np.random.default_rng(1)
    X_val = rng.uniform(-5, 5, (200, 2))
    y_val = np.array([bench.fn(X_val[i]) for i in range(200)])
    y_surr = surrogate.predict(X_val)

    print(f"Benchmark      : {bench.name}")
    print(f"Known optimum  : x={bench.x_opt}, f={bench.f_opt}")
    print(f"Surrogate found: x={np.round(result.x_best, 4)}, f_surrogate={result.f_best:.4f}")
    print(f"True f at x    : {result.f_true:.4f}")
    print(f"Gap to optimum : {abs(result.f_true - bench.f_opt):.4f}")
    print(f"Surrogate MAE  : {mae(y_val, y_surr):.4f}")
    print(f"Surrogate model: {result.surrogate.best_expression_}")


if __name__ == "__main__":
    main()
