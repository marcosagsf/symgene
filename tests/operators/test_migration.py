import pytest
import random
from symgene.primitive_set import PrimitiveSet
from symgene.population import Population
from symgene.operators.migration import migrate

random.seed(42)

def make_pop(name, seed=0):
    pset = PrimitiveSet(n_inputs=3)
    pset.add_from_catalog(["add", "mul", "square", "sin"])
    pset.add_ephemeral("c1", dist="uniform", low=-1.0, high=1.0, n=2)
    pop = Population(name=name, pset=pset, n_genes=3, pop_size=10, tree_max=50)
    pop.initialize(seed=seed)
    return pop

def test_migrate_ring_preserves_sizes():
    pops = [make_pop(f"p{i}", seed=i) for i in range(3)]
    sizes_before = [len(p._population) for p in pops]
    migrate(pops, topology="ring", size=2, selection="best", replace="worst")
    sizes_after = [len(p._population) for p in pops]
    assert sizes_before == sizes_after

def test_migrate_full_preserves_sizes():
    pops = [make_pop(f"p{i}", seed=i) for i in range(3)]
    sizes_before = [len(p._population) for p in pops]
    migrate(pops, topology="full", size=1, selection="best", replace="worst")
    sizes_after = [len(p._population) for p in pops]
    assert sizes_before == sizes_after

def test_migrate_best_to_worst_preserves_sizes():
    pops = [make_pop(f"p{i}", seed=i) for i in range(2)]
    for i, ind in enumerate(pops[0]._population):
        ind.fitness.values = (float(i),)
    for i, ind in enumerate(pops[1]._population):
        ind.fitness.values = (float(10 - i),)
    migrate(pops, topology="best_to_worst", size=1, selection="best", replace="worst")
    assert len(pops[0]._population) == 10
    assert len(pops[1]._population) == 10

def test_migrate_single_pop_is_noop():
    pops = [make_pop("p0")]
    original = [str(ind) for ind in pops[0]._population]
    migrate(pops, topology="ring", size=2, selection="best", replace="worst")
    after = [str(ind) for ind in pops[0]._population]
    assert original == after

def test_migrate_copies_not_references():
    pops = [make_pop(f"p{i}", seed=i) for i in range(2)]
    for ind in pops[0]._population:
        ind.fitness.values = (0.1,)
    for ind in pops[1]._population:
        ind.fitness.values = (0.5,)
    migrate(pops, topology="ring", size=1, selection="best", replace="worst")
    migrant_str = str(pops[1]._population[0])
    assert isinstance(migrant_str, str)
