"""
MGGP — Nguyen-10 & Double Well Potential (2D)
Multi-population MGGP on two 2D problems (x1, x2).

Library features showcased:
- regression_degree=2       — polynomial gene features
- RankSelection             — graded selection pressure
- missing_vars_penalty      — ensures all input variables appear in the expression
- schedule                  — adaptive parameter decay per generation
- EarlyStopping             — automatic stop on convergence

Run:
    python examples/04_mggp_nguyen10_doublewell.py
"""
import numpy as np
from sklearn.metrics import r2_score

from symgene import PrimitiveSet, Population, SymGeneEvolver
from symgene.callbacks import Callback, EarlyStopping, GenerationLogger
from symgene.fitness import FitnessEvaluator
from symgene.metrics import rmse, nrmse
from symgene.metrics.complexity import complexity_penalty, missing_vars_penalty
from symgene.metrics.regression import mse
from symgene.primitives import STANDARD
from symgene.selection import TournamentSelection
from symgene.selection.rank import RankSelection


class HistoryRecorder(Callback):
    def __init__(self):
        self.records = []

    def on_generation_end(self, gen, logs=None):
        if logs:
            self.records.append(dict(logs))
        return None


def nguyen10(x1, x2):
    return 2.0 * np.sin(x1) * np.cos(x2)


def doublewell(x1, x2):
    return x1**4 + x2**4 - 2.0 * (x1**2 + x2**2)


def lhs2d(rng, lo, hi, n):
    X = np.empty((n, 2))
    for j in range(2):
        perm = rng.permutation(n)
        X[:, j] = lo + (perm + rng.uniform(0, 1, n)) / n * (hi - lo)
    return X


def main():
    DOMAIN = (-1.5, 1.5)

    # ── Data generation (70 / 15 / 15 split) ────────────────────────────────
    rng = np.random.default_rng(42)
    N = 400
    X_all = lhs2d(rng, DOMAIN[0], DOMAIN[1], N)
    idx = rng.permutation(N)

    y_n_all = nguyen10(X_all[:, 0], X_all[:, 1])
    y_d_all = doublewell(X_all[:, 0], X_all[:, 1])

    n_train, n_val = 280, 60
    i_tr = idx[:n_train]
    i_v  = idx[n_train:n_train + n_val]
    i_te = idx[n_train + n_val:]

    X_train, X_val, X_test = X_all[i_tr], X_all[i_v], X_all[i_te]
    y_n_train, y_n_val, y_n_test = y_n_all[i_tr], y_n_all[i_v], y_n_all[i_te]
    y_d_train, y_d_val, y_d_test = y_d_all[i_tr], y_d_all[i_v], y_d_all[i_te]

    print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")
    print(f"Domain: x1, x2 in [{DOMAIN[0]}, {DOMAIN[1]}]")

    # ── PrimitiveSet & callbacks ─────────────────────────────────────────────
    history_cb = HistoryRecorder()

    pset = PrimitiveSet(n_inputs=2, feature_names=["x1", "x2"])
    pset.add_from_catalog(STANDARD)
    pset.set_squash(lim=10, alpha=0.08, scale=2.5)
    print(f"Primitives loaded: {len(pset.primitives)}")

    # ── Populations ──────────────────────────────────────────────────────────
    pop_nguyen = Population(
        name="nguyen10", pset=pset,
        n_genes=2, n_genes_max=6, pop_size=80,
        elite_ratio=0.05,
        tree_min=2, tree_max=25, tree_init_max=3, height_max=6,
        cxpb=0.85, cxpb_low=0.55,
        mutpb=0.35, mutpb_low=0.20,
        mutation_weights=[0.3, 1.8, 0.9],
        combiner="ridge",
        ridge_alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        regression_degree=2,
        fitness=FitnessEvaluator(metric=mse, penalties=[
            complexity_penalty(lambda_=3e-4),
        ]),
        selection=RankSelection(pressure=1.8),
        schedule={"mutpb": {0: 0.40, 60: 0.30, 100: 0.20}},
    )

    pop_dwell = Population(
        name="doublewell", pset=pset,
        n_genes=2, n_genes_max=10, pop_size=60,
        elite_ratio=0.03,
        tree_min=2, tree_max=35, tree_init_max=2, height_max=7,
        cxpb=0.90, cxpb_low=0.40,
        mutpb=0.35, mutpb_low=0.25,
        mutation_weights=[0.2, 2.0, 0.8],
        combiner="ridge",
        ridge_alphas=[0.01, 0.1, 1.0, 10.0, 100.0],
        regression_degree=1,
        fitness=FitnessEvaluator(metric=mse, penalties=[
            complexity_penalty(lambda_=1e-4),
            missing_vars_penalty(required={"x1", "x2"}, beta=0.05),
        ]),
        selection=TournamentSelection(size=7),
    )

    # ── Training ─────────────────────────────────────────────────────────────
    evolver = SymGeneEvolver(
        populations=[pop_nguyen, pop_dwell],
        n_gen=150,
        cross_population=False,
        seed=42,
        callbacks=[
            history_cb,
            GenerationLogger(every=30),
            EarlyStopping(monitor="doublewell_train_mse", patience=50, mode="min"),
        ],
        verbose=0,
    )

    results = evolver.fit(
        X_train,
        {"nguyen10": y_n_train, "doublewell": y_d_train},
        X_val=X_val,
        y_val={"nguyen10": y_n_val, "doublewell": y_d_val},
    )
    print("Training complete.")

    # ── Results ──────────────────────────────────────────────────────────────
    for pop_name, y_tr, y_v, y_te in [
        ("nguyen10",   y_n_train, y_n_val, y_n_test),
        ("doublewell", y_d_train, y_d_val, y_d_test),
    ]:
        res   = results[pop_name]
        yp_tr = res.predict(X_train)
        yp_v  = res.predict(X_val)
        yp_te = res.predict(X_test)
        print(f"\n{'=' * 60}")
        print(f"Population : {pop_name.upper()}")
        print(f"Genes      : {res.n_genes_}")
        print(f"Expression : {res.best_expression_}")
        print(f"Train  RMSE={rmse(y_tr, yp_tr):.4f}  NRMSE={nrmse(y_tr, yp_tr):.4f}  R²={r2_score(y_tr, yp_tr):.4f}")
        print(f"Val    RMSE={rmse(y_v,  yp_v ):.4f}  NRMSE={nrmse(y_v,  yp_v ):.4f}  R²={r2_score(y_v,  yp_v ):.4f}")
        print(f"Test   RMSE={rmse(y_te, yp_te):.4f}  NRMSE={nrmse(y_te, yp_te):.4f}  R²={r2_score(y_te, yp_te):.4f}")


if __name__ == "__main__":
    main()
