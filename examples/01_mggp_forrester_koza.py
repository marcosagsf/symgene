"""
MGGP — Forrester 1D & Koza-1
Multi-population symbolic regression using SymGeneEvolver with two simultaneous
populations on 1D analytic benchmark functions.

Run:
    python examples/01_mggp_forrester_koza.py
"""
import numpy as np
from sklearn.metrics import r2_score

from symgene import PrimitiveSet, Population, SymGeneEvolver
from symgene.benchmarks import forrester_1d
from symgene.callbacks import Callback, GenerationLogger
from symgene.fitness import FitnessEvaluator
from symgene.metrics import rmse, nrmse
from symgene.metrics.complexity import complexity_penalty
from symgene.metrics.regression import mse
from symgene.primitives import STANDARD
from symgene.selection import TournamentSelection


class HistoryRecorder(Callback):
    def __init__(self):
        self.records = []

    def on_generation_end(self, gen, logs=None):
        if logs:
            self.records.append(dict(logs))
        return None


def main():
    # ── Data generation (70 / 15 / 15 split) ────────────────────────────────
    bench = forrester_1d()
    koza_fn = lambda x: x**4 + x**3 + x**2 + x

    rng = np.random.default_rng(0)
    X_all = rng.uniform(0, 1, (200, 1))
    idx = rng.permutation(200)
    y_f_all = np.array([bench.fn(X_all[i]) for i in range(200)])
    y_k_all = koza_fn(X_all[:, 0])

    n_train, n_val = 140, 30
    i_tr, i_v, i_te = idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]

    X_train, X_val, X_test = X_all[i_tr], X_all[i_v], X_all[i_te]
    y_f_train, y_f_val, y_f_test = y_f_all[i_tr], y_f_all[i_v], y_f_all[i_te]
    y_k_train, y_k_val, y_k_test = y_k_all[i_tr], y_k_all[i_v], y_k_all[i_te]

    print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

    # ── Model setup ──────────────────────────────────────────────────────────
    history_cb = HistoryRecorder()

    pset = PrimitiveSet(n_inputs=1, feature_names=["x"])
    pset.add_from_catalog(STANDARD)
    pset.set_squash(lim=8, alpha=0.1, scale=2.0)
    pset.add_custom(
        fn=lambda x: float(np.exp(-((x - 0.5) ** 2) / 0.1)),
        arity=1,
        name="bump",
    )

    pop_forrester = Population(
        name="forrester", pset=pset,
        n_genes=1, n_genes_max=10, pop_size=50,
        elite_ratio=0.025,
        tree_min=2, tree_max=40, tree_init_max=2, height_max=8,
        cxpb=0.90, cxpb_low=0.40,
        mutpb=0.35, mutpb_low=0.25,
        mutation_weights=[0.2, 2.0, 0.8],
        fitness=FitnessEvaluator(metric=mse, penalties=[complexity_penalty(lambda_=1e-4)]),
        selection=TournamentSelection(size=2),
    )
    pop_koza = Population(
        name="koza1", pset=pset,
        n_genes=1, n_genes_max=8, pop_size=50,
        elite_ratio=0.025,
        tree_min=2, tree_max=30, tree_init_max=2, height_max=8,
        cxpb=0.90, cxpb_low=0.50,
        mutpb=0.35, mutpb_low=0.25,
        mutation_weights=[0.2, 2.0, 0.8],
        fitness=FitnessEvaluator(metric=mse, penalties=[complexity_penalty(lambda_=5e-4)]),
        selection=TournamentSelection(size=5),
    )

    evolver = SymGeneEvolver(
        populations=[pop_forrester, pop_koza], n_gen=300,
        cross_population=False, seed=0,
        callbacks=[history_cb, GenerationLogger(every=20)],
        verbose=0,
    )

    # ── Training ─────────────────────────────────────────────────────────────
    results = evolver.fit(
        X_train,
        {"forrester": y_f_train, "koza1": y_k_train},
        X_val=X_val,
        y_val={"forrester": y_f_val, "koza1": y_k_val},
    )
    print("Training complete.")

    # ── Results ──────────────────────────────────────────────────────────────
    for pop_name, y_tr, y_v, y_te in [
        ("forrester", y_f_train, y_f_val, y_f_test),
        ("koza1",     y_k_train, y_k_val, y_k_test),
    ]:
        res = results[pop_name]
        yp_tr = res.predict(X_train)
        yp_v  = res.predict(X_val)
        yp_te = res.predict(X_test)
        print(f"\n{'=' * 55}")
        print(f"Population : {pop_name.upper()}")
        print(f"Genes      : {res.n_genes_}")
        print(f"Expression : {res.best_expression_}")
        print(f"Train  RMSE={rmse(y_tr, yp_tr):.4f}  NRMSE={nrmse(y_tr, yp_tr):.4f}  R²={r2_score(y_tr, yp_tr):.4f}")
        print(f"Val    RMSE={rmse(y_v,  yp_v ):.4f}  NRMSE={nrmse(y_v,  yp_v ):.4f}  R²={r2_score(y_v,  yp_v ):.4f}")
        print(f"Test   RMSE={rmse(y_te, yp_te):.4f}  NRMSE={nrmse(y_te, yp_te):.4f}  R²={r2_score(y_te, yp_te):.4f}")


if __name__ == "__main__":
    main()
