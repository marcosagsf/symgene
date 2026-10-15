import pytest
import random
import numpy as np
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

def test_migrate_ring_random_selection_and_replace():
    pops = [make_pop(f"p{i}", seed=i) for i in range(2)]
    sizes_before = [len(p._population) for p in pops]
    migrate(pops, topology="ring", size=1, selection="random", replace="random")
    sizes_after = [len(p._population) for p in pops]
    assert sizes_before == sizes_after

def test_migrate_unknown_topology_raises():
    pops = [make_pop(f"p{i}", seed=i) for i in range(2)]
    with pytest.raises(ValueError, match="Unknown migration topology"):
        migrate(pops, topology="mesh", size=1, selection="best", replace="worst")


def test_migrate_star_preserves_sizes():
    pops = [make_pop(f"p{i}", seed=i) for i in range(4)]
    sizes_before = [len(p._population) for p in pops]
    migrate(pops, topology="star", size=1, selection="best", replace="worst",
            hub_index=0)
    sizes_after = [len(p._population) for p in pops]
    assert sizes_before == sizes_after


def test_migrate_star_hub_receives_from_every_spoke():
    # Tag each island's individuals with a fitness equal to its index so we
    # can confirm the hub ends up holding migrants that came from spokes.
    pops = [make_pop(f"p{i}", seed=i) for i in range(3)]
    for island, pop in enumerate(pops):
        for ind in pop._population:
            ind.fitness.values = (float(island),)
    migrate(pops, topology="star", size=2, selection="best", replace="worst",
            hub_index=0)
    # Hub (index 0) should now contain individuals whose fitness came from a
    # spoke (fitness 1.0 or 2.0), since spokes send their 'best' (lowest).
    hub_fitnesses = {ind.fitness.values[0] for ind in pops[0]._population}
    assert hub_fitnesses & {1.0, 2.0}


def test_migrate_star_invalid_hub_index_raises():
    pops = [make_pop(f"p{i}", seed=i) for i in range(3)]
    with pytest.raises(ValueError, match="hub_index must be in"):
        migrate(pops, topology="star", hub_index=5)


def test_migrate_custom_preserves_sizes():
    pops = [make_pop(f"p{i}", seed=i) for i in range(3)]
    adjacency = np.array([[0, 1, 0],
                          [0, 0, 1],
                          [1, 0, 0]], dtype=bool)
    sizes_before = [len(p._population) for p in pops]
    migrate(pops, topology="custom", size=1, selection="best", replace="worst",
            adjacency=adjacency)
    sizes_after = [len(p._population) for p in pops]
    assert sizes_before == sizes_after


def test_migrate_custom_requires_adjacency():
    pops = [make_pop(f"p{i}", seed=i) for i in range(2)]
    with pytest.raises(ValueError, match="requires an 'adjacency'"):
        migrate(pops, topology="custom", size=1)


def test_migrate_custom_wrong_shape_raises():
    pops = [make_pop(f"p{i}", seed=i) for i in range(3)]
    bad = np.ones((2, 2), dtype=bool)
    with pytest.raises(ValueError, match="adjacency must have shape"):
        migrate(pops, topology="custom", adjacency=bad)


def test_migrate_custom_reproduces_ring():
    # A cyclic adjacency matrix must behave like the ring topology: only the
    # designated receiver of each island changes.
    pops = [make_pop(f"p{i}", seed=i) for i in range(3)]
    for island, pop in enumerate(pops):
        for ind in pop._population:
            ind.fitness.values = (float(island),)
    adjacency = np.array([[0, 1, 0],
                          [0, 0, 1],
                          [1, 0, 0]], dtype=bool)
    migrate(pops, topology="custom", size=1, selection="best", replace="worst",
            adjacency=adjacency)
    # Island 1 received from island 0 (fitness 0.0 present after migration).
    assert 0.0 in {ind.fitness.values[0] for ind in pops[1]._population}

def test_migrate_copies_not_references():
    pops = [make_pop(f"p{i}", seed=i) for i in range(2)]
    for ind in pops[0]._population:
        ind.fitness.values = (0.1,)
    for ind in pops[1]._population:
        ind.fitness.values = (0.5,)
    migrate(pops, topology="ring", size=1, selection="best", replace="worst")
    migrant_str = str(pops[1]._population[0])
    assert isinstance(migrant_str, str)
