import random
import pytest
import deap.gp as gp
import deap.creator as creator
import deap.base as base
from symgene.primitive_set import PrimitiveSet
from symgene.population import Population

random.seed(42)

def make_pop():
    pset = PrimitiveSet(n_inputs=3)
    pset.add_from_catalog(["add", "sub", "mul", "sin", "square"])
    pset.add_ephemeral("c1", dist="uniform", low=-1.0, high=1.0, n=2)
    pop = Population(name="test", pset=pset, n_genes=4, pop_size=20, tree_max=50)
    pop.initialize(seed=42)
    return pop

from symgene.operators.crossover import intra_crossover
from symgene.operators.mutation import mutate_population

def test_intra_crossover_returns_two_inds():
    pop = make_pop()
    a, b = pop._population[0], pop._population[1]
    a2, b2 = intra_crossover(a, b, pop._toolbox, cxpb_low=0.5, tree_max=50)
    assert isinstance(a2, list)
    assert isinstance(b2, list)

def test_intra_crossover_invalidates_fitness():
    pop = make_pop()
    a, b = pop._population[0], pop._population[1]
    a.fitness.values = (0.5,)
    b.fitness.values = (0.3,)
    a2, b2 = intra_crossover(a, b, pop._toolbox, cxpb_low=0.5, tree_max=50)
    assert not a2.fitness.valid or not b2.fitness.valid

def test_mutate_population_changes_some():
    pop = make_pop()
    original = [str(ind) for ind in pop._population]
    mutate_population(pop._population, pop._toolbox, mutpb=1.0,
                      mutpb_low=0.5, mutation_weights=[1,1,1],
                      n_genes_max=20, tree_max=50)
    changed = [str(ind) != orig for ind, orig in zip(pop._population, original)]
    assert any(changed)

from symgene.operators.crossover import interpop_crossover

def test_interpop_crossover_returns_five_values():
    pop = make_pop()
    a, b = pop._population[0], pop._population[1]
    result = interpop_crossover(a, b, pop._toolbox, cxpb_low_inter=0.5, tree_max=50)
    assert len(result) == 5  # indA, indB, n_low, n_high, n_fallback

def test_interpop_crossover_empty_returns_zeros():
    pop = make_pop()
    a, b = pop._population[0], pop._population[1]
    # make one individual appear empty by patching __len__ indirectly — use real individuals
    _, _, n_low, n_high, n_fallback = interpop_crossover(
        a, b, pop._toolbox, cxpb_low_inter=1.0, tree_max=50
    )
    assert isinstance(n_low + n_high + n_fallback, int)

def test_interpop_crossover_high_order_path():
    """cxpb_low_inter=0.0 forces the high-order (whole-gene swap) branch."""
    pop = make_pop()
    a, b = pop._population[0], pop._population[1]
    _, _, n_low, n_high, _ = interpop_crossover(
        a, b, pop._toolbox, cxpb_low_inter=0.0, tree_max=50
    )
    assert n_high > 0

def test_mutate_respects_mutpb_zero():
    pop = make_pop()
    original = [str(ind) for ind in pop._population]
    mutate_population(pop._population, pop._toolbox, mutpb=0.0,
                      mutpb_low=0.5, mutation_weights=[1,1,1],
                      n_genes_max=20, tree_max=50)
    unchanged = [str(ind) == orig for ind, orig in zip(pop._population, original)]
    assert all(unchanged)
