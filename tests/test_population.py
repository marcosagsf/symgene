import numpy as np
import pytest
from symgene.primitive_set import PrimitiveSet
from symgene.primitives.catalog import STANDARD
from symgene.population import Population
from symgene.fitness import FitnessEvaluator
from symgene.metrics.regression import mse

rng = np.random.default_rng(42)
X = rng.standard_normal((50, 3))
y = X[:, 0] ** 2 - X[:, 1]

def make_pset():
    pset = PrimitiveSet(n_inputs=3, feature_names=["a", "b", "c"])
    pset.add_from_catalog(["add", "mul", "square", "sin"])
    pset.add_ephemeral("c1", dist="uniform", low=-1.0, high=1.0, n=2)
    pset.set_squash(lim=8)
    return pset

def test_population_builds_deap_objects():
    pop = Population(name="test", pset=make_pset(), n_genes=3, pop_size=10)
    pop.initialize()
    assert len(pop._population) == 10

def test_population_individual_has_n_genes():
    pop = Population(name="test", pset=make_pset(), n_genes=3, pop_size=10)
    pop.initialize()
    for ind in pop._population:
        assert len(ind) == 3

def test_population_evaluate_sets_fitness():
    ev = FitnessEvaluator(metric=mse)
    pop = Population(name="test", pset=make_pset(), n_genes=3, pop_size=10,
                     fitness=ev, combiner="ridge")
    pop.initialize()
    pop.evaluate(X, y)
    for ind in pop._population:
        assert ind.fitness.valid

def test_population_schedule_applies():
    pop = Population(
        name="test", pset=make_pset(), n_genes=3, pop_size=10,
        tree_max=80,
        schedule={"tree_max": {0: 80, 5: 120}},
    )
    pop.initialize()
    pop.apply_schedule(generation=0)
    assert pop.tree_max == 80
    pop.apply_schedule(generation=5)
    assert pop.tree_max == 120
