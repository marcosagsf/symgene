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

def test_mutate_respects_mutpb_zero():
    pop = make_pop()
    original = [str(ind) for ind in pop._population]
    mutate_population(pop._population, pop._toolbox, mutpb=0.0,
                      mutpb_low=0.5, mutation_weights=[1,1,1],
                      n_genes_max=20, tree_max=50)
    unchanged = [str(ind) == orig for ind, orig in zip(pop._population, original)]
    assert all(unchanged)
