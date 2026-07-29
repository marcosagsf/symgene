"""
Medicine BMD Demo — SymGene
Demonstrates SymGeneRegressor for single-output regression on a synthetic
BMD (Bone Mineral Density) dataset with 8 clinical features.
Run: python -m symgene.examples.medicine_bmd.run
"""
import numpy as np

from symgene import SymGeneRegressor
from symgene.primitives import STANDARD


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
    )
    model.fit(X_tr, y_tr, X_val=X_te, y_val=y_te)

    print(f"Genes     : {model.n_genes_}")
    print(f"Fitness   : {model.best_fitness_:.6f}")
    print(f"Expression: {model.best_expression_}")
    print(f"LaTeX     : {model.to_latex()}")

    y_pred = model.predict(X_te)
    from symgene.metrics.regression import r2
    print(f"Test R²   : {r2(y_te, y_pred):.4f}")


if __name__ == "__main__":
    main()
