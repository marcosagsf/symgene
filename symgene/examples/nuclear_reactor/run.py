"""
Nuclear Reactor MGGP Demo — SymGene
Mimics the Angra II reactor methodology: two populations (CBORON + PPF)
with 108 synthetic inputs (36 k-inf + 36 LEX + 36 cluster).
Run: python -m symgene.examples.nuclear_reactor.run
"""
import numpy as np

from symgene import PrimitiveSet, Population, SymGeneEvolver
from symgene.primitives import STANDARD
from symgene.fitness import FitnessEvaluator
from symgene.metrics.regression import mse
from symgene.metrics.complexity import complexity_penalty, missing_vars_penalty
from symgene.selection import TournamentSelection
from symgene.callbacks import EarlyStopping, GenerationLogger


def make_data(n_train=300, n_test=100, seed=42):
    rng = np.random.default_rng(seed)
    kinf = rng.uniform(0.85, 1.15, (n_train + n_test, 36))
    lex = kinf - kinf.mean(axis=1, keepdims=True) * rng.uniform(0.95, 1.05, kinf.shape)
    cluster = kinf * kinf.mean(axis=1, keepdims=True)
    X = np.hstack([kinf, lex, cluster])
    y_cboron = kinf.sum(axis=1) * 50.0 + rng.normal(0, 0.5, n_train + n_test)
    y_ppf = (
        np.sin(kinf[:, 0] * np.pi) * np.cos(kinf[:, 1] * np.pi)
        + kinf[:, :4].mean(axis=1)
        + rng.normal(0, 0.02, n_train + n_test)
    )
    X_tr, X_te = X[:n_train], X[n_train:]
    return X_tr, X_te, y_cboron[:n_train], y_cboron[n_train:], y_ppf[:n_train], y_ppf[n_train:]


def make_pset():
    feature_names = (
        [f"kinf_{i+1}" for i in range(36)]
        + [f"lex_{i+1}" for i in range(36)]
        + [f"cluster_{i+1}" for i in range(36)]
    )
    pset = PrimitiveSet(n_inputs=108, feature_names=feature_names)
    pset.add_from_catalog(STANDARD)
    pset.set_squash(lim=8, alpha=0.1, scale=2.0)

    def local_peak(center, n1, n2, n3, n4):
        return center - (n1 + n2 + n3 + n4) / 4.0

    pset.add_custom(local_peak, arity=5, name="local_peak")
    return pset


def main():
    X_tr, X_te, y_cb_tr, y_cb_te, y_ppf_tr, y_ppf_te = make_data()
    pset = make_pset()

    pop_cboron = Population(
        name="cboron",
        pset=pset,
        n_genes=4,
        pop_size=30,
        fitness=FitnessEvaluator(
            metric=mse,
            penalties=[
                complexity_penalty(lambda_=5e-4),
                missing_vars_penalty(
                    required={f"kinf_{i+1}" for i in range(36)}, beta=0.01
                ),
            ],
        ),
        selection=TournamentSelection(size=5),
    )

    pop_ppf = Population(
        name="ppf",
        pset=pset,
        n_genes=4,
        pop_size=30,
        fitness=FitnessEvaluator(metric=mse, penalties=[complexity_penalty(lambda_=3e-4)]),
        selection=TournamentSelection(size=5),
    )

    evolver = SymGeneEvolver(
        populations=[pop_cboron, pop_ppf],
        n_gen=15,
        cross_population=True,
        cxpb_inter=0.025,
        seed=0,
        callbacks=[
            GenerationLogger(every=5),
            EarlyStopping(monitor="cboron_train_mse", patience=5),
        ],
        verbose=0,
    )

    results = evolver.fit(
        X_tr,
        {"cboron": y_cb_tr, "ppf": y_ppf_tr},
        X_val=X_te,
        y_val={"cboron": y_cb_te, "ppf": y_ppf_te},
    )

    for pop_name in ("cboron", "ppf"):
        res = results[pop_name]
        print(f"\n=== {pop_name.upper()} ===")
        print(f"  Genes  : {res.n_genes_}")
        print(f"  Fitness: {res.best_fitness_:.6f}")

    print("\nDone. To export: results['cboron'].to_latex()")


if __name__ == "__main__":
    main()
