"""
Forrester 1D & Koza-1 — Multi-population MGGP Demo

Demonstrates SymGeneEvolver with two simultaneous populations over
1D analytic functions. Since the functions are known analytically, we generate
as many points as needed.

Demonstrated parameters:
  n_genes_max      — maximum number of genes per individual (add mutation respects this limit)
  elite_ratio      — fraction of the population preserved intact each generation (elitism)
  tree_min         — minimum nodes per GP tree
  tree_max         — maximum nodes per GP tree (controls bloat)
  height_max       — maximum nesting depth
  cxpb             — probability of crossover occurring per generation
  cxpb_low         — fraction low-order (subtree) vs high-order (whole gene swap)
  cross_population — enables gene exchange between distinct populations
  cxpb_inter       — probability of inter-population crossover per generation
  FitnessEvaluator — fitness with built-in complexity penalty
  complexity_penalty — penalizes models with many nodes
  EarlyStopping    — stops evolution when there is no improvement
  GenerationLogger — prints metrics every N generations
  add_custom       — user-defined mathematical primitive

Run: python -m symgene.examples.forrester_mggp.run
"""
import numpy as np

from symgene import PrimitiveSet, Population, SymGeneEvolver
from symgene.primitives import STANDARD
from symgene.fitness import FitnessEvaluator
from symgene.metrics.regression import mse
from symgene.metrics.complexity import complexity_penalty
from symgene.metrics import rmse, nrmse
from symgene.selection import TournamentSelection
from symgene.callbacks import EarlyStopping, GenerationLogger
from symgene.benchmarks import forrester_1d


def main():
    bench = forrester_1d()

    # Both functions evaluated on the same domain [0, 1]
    rng = np.random.default_rng(0)
    X_train = np.sort(rng.uniform(0, 1, (200, 1)), axis=0)
    X_test  = np.linspace(0, 1, 500).reshape(-1, 1)

    y_f_train = np.array([bench.fn(X_train[i]) for i in range(200)])
    y_f_test  = np.array([bench.fn(X_test[i])  for i in range(500)])

    # Koza-1: x^4 + x^3 + x^2 + x — evaluated on the same X
    koza_fn   = lambda x: x**4 + x**3 + x**2 + x
    y_k_train = koza_fn(X_train[:, 0])
    y_k_test  = koza_fn(X_test[:, 0])

    # PrimitiveSet shared between the two populations
    pset = PrimitiveSet(n_inputs=1, feature_names=["x"])
    pset.add_from_catalog(STANDARD)
    pset.set_squash(lim=8, alpha=0.1, scale=2.0)
    # custom primitive: Gaussian bump centered at 0.5
    # demonstrates how to add domain-specific functions
    pset.add_custom(
        fn=lambda x: float(np.exp(-((x - 0.5) ** 2) / 0.1)),
        arity=1,
        name="bump",
    )

    pop_forrester = Population(
        name="forrester",
        pset=pset,
        n_genes=4,
        n_genes_max=10,       # allows growing up to 10 genes via add mutation
        pop_size=50,
        elite_ratio=0.05,     # 5% of the population preserved intact (elitism)
        tree_min=10,          # trees with at least 10 nodes
        tree_max=40,          # maximum of 40 nodes per tree (avoids bloat)
        height_max=5,         # maximum of 5 nesting levels
        cxpb=0.90,            # 90% chance of crossover per generation
        cxpb_low=0.40,        # 40% subtree (low), 60% whole gene swap (high)
        fitness=FitnessEvaluator(
            metric=mse,
            penalties=[complexity_penalty(lambda_=1e-4)],  # penalizes complex models
        ),
        selection=TournamentSelection(size=5),
    )

    pop_koza = Population(
        name="koza1",
        pset=pset,
        n_genes=4,
        n_genes_max=8,
        pop_size=50,
        elite_ratio=0.05,
        tree_min=5,
        tree_max=30,
        height_max=4,
        cxpb=0.85,
        cxpb_low=0.50,
        fitness=FitnessEvaluator(
            metric=mse,
            penalties=[complexity_penalty(lambda_=5e-4)],
        ),
        selection=TournamentSelection(size=5),
    )

    evolver = SymGeneEvolver(
        populations=[pop_forrester, pop_koza],
        n_gen=80,
        cross_population=True,   # genes migrate between the two populations
        cxpb_inter=0.025,        # 2.5% chance of inter-population crossover
        seed=0,
        callbacks=[
            GenerationLogger(every=20),
            EarlyStopping(monitor="forrester_train_mse", patience=15),
        ],
        verbose=0,
    )

    results = evolver.fit(
        X_train,
        {"forrester": y_f_train, "koza1": y_k_train},
        X_val=X_test,
        y_val={"forrester": y_f_test, "koza1": y_k_test},
    )

    for pop_name, y_te in [("forrester", y_f_test), ("koza1", y_k_test)]:
        res = results[pop_name]
        y_pred = res.predict(X_test)
        print(f"\n=== {pop_name.upper()} ===")
        print(f"  Genes     : {res.n_genes_}")
        print(f"  Expression: {res.best_expression_}")
        print(f"  RMSE      : {rmse(y_te, y_pred):.4f}")
        print(f"  NRMSE     : {nrmse(y_te, y_pred):.4f}")


if __name__ == "__main__":
    main()
