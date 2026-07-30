"""
Medicine BMD Demo — SymGene
Demonstra SymGeneRegressor para regressão de um único output em um dataset
sintético de BMD (Bone Mineral Density) com 8 variáveis clínicas.

Parâmetros demonstrados:
  combiner          — combinador linear dos genes: 'ridge' (padrão), 'lasso' ou 'linear'
  ridge_alphas      — lista de alphas testados na validação cruzada interna do Ridge
                      alpha alto → modelo mais suave; alpha baixo → mais ajustado
  regression_degree — grau do modelo de regressão sobre os genes
                      1 = linear (padrão), 2 = quadrático (interage pares de genes)

Run: python -m symgene.examples.medicine_bmd.run
"""
import numpy as np

from symgene import SymGeneRegressor
from symgene.primitives import STANDARD
from symgene.metrics import r2, mape


FEATURE_NAMES = ["age", "bmi", "bmd_prev", "activity", "calcium", "vitamin_d", "sex", "fracture_hist"]


def make_data(n=250, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n, 8))
    y = (
        0.80 * X[:, 2]
        - 0.10 * X[:, 0]
        + 0.05 * X[:, 3]
        + 0.03 * X[:, 4]
        + rng.normal(0, 0.02, n)
    )
    n_tr = int(0.8 * n)
    return X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:]


def main():
    X_tr, y_tr, X_te, y_te = make_data()

    model = SymGeneRegressor(
        n_genes=4,
        pop_size=30,
        n_gen=15,
        primitives=STANDARD,
        squash={"lim": 8, "alpha": 0.1, "scale": 2.0},
        feature_names=FEATURE_NAMES,
        seed=0,
        verbose=0,
        # --- combinador e regularização ---
        combiner="ridge",                        # Ridge regulariza os pesos dos genes
        ridge_alphas=[0.01, 0.1, 1.0, 10.0, 100.0],  # alphas testados internamente
        regression_degree=1,                     # combinação linear dos genes (grau 1)
    )
    model.fit(X_tr, y_tr, X_val=X_te, y_val=y_te)

    y_pred = model.predict(X_te)

    print(f"Genes     : {model.n_genes_}")
    print(f"Fitness   : {model.best_fitness_:.6f}")
    print(f"Expression: {model.best_expression_}")
    print(f"LaTeX     : {model.to_latex()}")
    print(f"Test R²   : {r2(y_te, y_pred):.4f}")
    print(f"Test MAPE : {mape(y_te, y_pred):.2f}%")


if __name__ == "__main__":
    main()
