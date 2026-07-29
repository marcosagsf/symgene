import numpy as np
import pytest
from symgene.selection.tournament import TournamentSelection

class FakeInd:
    def __init__(self, fitness):
        self.fitness_value = fitness
    def clone(self): return FakeInd(self.fitness_value)

def make_pop(fitnesses):
    return [FakeInd(f) for f in fitnesses]

def test_tournament_selects_correct_count():
    pop = make_pop([0.5, 0.3, 0.8, 0.1, 0.6])
    sel = TournamentSelection(size=3)
    selected = sel.select(pop, k=3, fitness_attr="fitness_value", minimize=True)
    assert len(selected) == 3

def test_tournament_tends_to_select_best():
    rng = np.random.default_rng(42)
    pop = make_pop([float(i) for i in range(100)])
    sel = TournamentSelection(size=10)
    selected = sel.select(pop, k=50, fitness_attr="fitness_value", minimize=True)
    avg_fitness = np.mean([s.fitness_value for s in selected])
    assert avg_fitness < 50.0

from symgene.selection.gene_selection import select_genes_by_weight

def test_gene_selection_returns_k_indices():
    weights = np.array([0.1, 0.5, 0.3, 0.0, 0.8])
    idxs = select_genes_by_weight(weights, k=3)
    assert len(idxs) == 3
    assert len(set(idxs)) == 3

def test_gene_selection_avoids_zero_weight():
    weights = np.array([0.0, 1.0, 0.0, 1.0])
    for _ in range(20):
        idxs = select_genes_by_weight(weights, k=2)
        assert 0 not in idxs
        assert 2 not in idxs

def test_gene_selection_fallback_uniform():
    weights = np.array([0.0, 0.0, 0.0])
    idxs = select_genes_by_weight(weights, k=2)
    assert len(idxs) == 2
