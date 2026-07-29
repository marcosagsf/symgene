import numpy as np
import pytest
from symgene.primitive_set import PrimitiveSet
from symgene.population import Population
from symgene.evolver import SymGeneEvolver
from symgene.fitness import FitnessEvaluator
from symgene.metrics.regression import mse

rng = np.random.default_rng(0)
X = rng.standard_normal((80, 3))
y = X[:, 0] ** 2 - X[:, 1] + 0.5

def make_pset():
    pset = PrimitiveSet(n_inputs=3)
    pset.add_from_catalog(["add", "sub", "mul", "square", "sin"])
    pset.set_squash(lim=8)
    return pset

def make_pop(name="p1"):
    return Population(
        name=name, pset=make_pset(), n_genes=3, pop_size=20,
        fitness=FitnessEvaluator(metric=mse), combiner="ridge",
    )

def test_evolver_single_pop_fit_returns_result():
    evolver = SymGeneEvolver(populations=[make_pop()], n_gen=3, seed=0)
    results = evolver.fit(X, {"p1": y})
    assert "p1" in results

def test_evolver_result_has_best():
    evolver = SymGeneEvolver(populations=[make_pop()], n_gen=3, seed=0)
    results = evolver.fit(X, {"p1": y})
    assert results["p1"].best_individual_ is not None

def test_evolver_result_predict():
    evolver = SymGeneEvolver(populations=[make_pop()], n_gen=3, seed=0)
    results = evolver.fit(X, {"p1": y})
    y_pred = results["p1"].predict(X)
    assert y_pred.shape == (80,)

def test_evolver_history_grows():
    evolver = SymGeneEvolver(populations=[make_pop()], n_gen=5, seed=0)
    results = evolver.fit(X, {"p1": y})
    assert len(results["p1"].history_) == 5

def test_evolver_with_val():
    X_val = rng.standard_normal((20, 3))
    y_val = X_val[:, 0] ** 2 - X_val[:, 1] + 0.5
    evolver = SymGeneEvolver(populations=[make_pop()], n_gen=3, seed=0)
    results = evolver.fit(X, {"p1": y}, X_val=X_val, y_val={"p1": y_val})
    hist = results["p1"].history_
    assert "val_r2" in hist[0]

def test_evolver_migration_preserves_pop_sizes():
    pop1 = make_pop("p1")
    pop2 = make_pop("p2")
    evolver = SymGeneEvolver(
        populations=[pop1, pop2], n_gen=5, seed=0,
        migration=True, migration_freq=2, migration_size=1,
        migration_topology="ring", migration_selection="best",
        migration_replace="worst", verbose=0,
    )
    y2 = rng.standard_normal(80)
    results = evolver.fit(X, {"p1": y, "p2": y2})
    assert len(pop1._population) == 20
    assert len(pop2._population) == 20

def test_evolver_migration_disabled_by_default():
    pop1 = make_pop("p1")
    pop2 = make_pop("p2")
    evolver = SymGeneEvolver(
        populations=[pop1, pop2], n_gen=3, seed=0, verbose=0
    )
    y2 = rng.standard_normal(80)
    results = evolver.fit(X, {"p1": y, "p2": y2})
    assert "p1" in results and "p2" in results
