"""
Forrester 1D — MGGP Surrogate Demo
Fits a SymGeneRegressor surrogate on the Forrester function and compares
the surrogate expression with the known analytical formula.

Parâmetros demonstrados:
  tree_min, tree_max  — controle do tamanho das árvores GP (em nós)
  height_max          — altura máxima da árvore
  cxpb                — probabilidade de crossover ocorrer por geração
  cxpb_low            — fração do crossover que é low-order (subtree)
                        vs high-order (troca inteira de gene)

Run: python -m symgene.examples.forrester_mggp.run
"""
import numpy as np
from symgene import SymGeneRegressor
from symgene.primitives import STANDARD
from symgene.benchmarks import forrester_1d
from symgene.metrics import rmse, r2


def main():
    bench = forrester_1d()
    rng = np.random.default_rng(0)

    # Função analítica conhecida — podemos gerar quantos pontos quisermos
    X_train = np.sort(rng.uniform(0, 1, (200, 1)), axis=0)
    y_train = np.array([bench.fn(X_train[i]) for i in range(200)])

    X_test = np.linspace(0, 1, 500).reshape(-1, 1)
    y_test = np.array([bench.fn(X_test[i]) for i in range(500)])

    model = SymGeneRegressor(
        n_genes=4,
        pop_size=60,
        n_gen=80,
        primitives=STANDARD,
        squash={"lim": 8, "alpha": 0.1, "scale": 2.0},
        feature_names=["x"],
        seed=0,
        verbose=0,
        # --- controle de árvore ---
        tree_min=10,    # mínimo de nós por árvore (árvores menores = expressões mais simples)
        tree_max=40,    # máximo de nós por árvore (evita bloat)
        height_max=5,   # altura máxima (limita profundidade de aninhamento)
        # --- controle de crossover ---
        cxpb=0.90,      # 90% das gerações realizam crossover
        cxpb_low=0.40,  # 40% low-order (subtree), 60% high-order (troca de gene inteiro)
    )
    model.fit(X_train, y_train, X_val=X_test, y_val=y_test)

    y_pred = model.predict(X_test)

    print(f"Benchmark : {bench.name}")
    print(f"Formula   : {bench.formula}")
    print(f"Genes     : {model.n_genes_}")
    print(f"Expression: {model.best_expression_}")
    print(f"LaTeX     : {model.to_latex()}")
    print(f"Test RMSE : {rmse(y_test, y_pred):.4f}")
    print(f"Test R²   : {r2(y_test, y_pred):.4f}")


if __name__ == "__main__":
    main()
