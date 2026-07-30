"""
Forrester 1D — MGGP Surrogate Demo
Fits a SymGeneRegressor surrogate on the Forrester function,
then compares the surrogate expression with the known formula.
Run: python -m symgene.examples.forrester_mggp.run
"""
import numpy as np
from symgene import SymGeneRegressor
from symgene.primitives import STANDARD
from symgene.benchmarks import forrester_1d


def main():
    bench = forrester_1d()
    rng = np.random.default_rng(0)

    # Training: 200 uniform points in [0, 1] — function is known analytically
    X_train = np.sort(rng.uniform(0, 1, (200, 1)), axis=0)
    y_train = np.array([bench.fn(X_train[i]) for i in range(200)])

    # Test: 500 evenly spaced points
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
    )
    model.fit(X_train, y_train, X_val=X_test, y_val=y_test)

    y_pred = model.predict(X_test)
    rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
    corr = float(np.corrcoef(y_test, y_pred)[0, 1])

    print(f"Benchmark : {bench.name}")
    print(f"Formula   : {bench.formula}")
    print(f"Genes     : {model.n_genes_}")
    print(f"Expression: {model.best_expression_}")
    print(f"LaTeX     : {model.to_latex()}")
    print(f"Test RMSE : {rmse:.4f}")
    print(f"Corr(true,surrogate): {corr:.4f}")


if __name__ == "__main__":
    main()
