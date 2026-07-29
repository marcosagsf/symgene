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

from symgene.selection.roulette import RouletteSelection
from symgene.selection.rank import RankSelection


def test_roulette_returns_k():
    pop = make_pop([0.5, 0.3, 0.8, 0.1, 0.6])
    sel = RouletteSelection()
    selected = sel.select(pop, k=4, fitness_attr="fitness_value", minimize=True)
    assert len(selected) == 4


def test_roulette_tends_to_select_best():
    rng = np.random.default_rng(99)
    pop = make_pop([float(i) for i in range(100)])
    sel = RouletteSelection()
    selected = sel.select(pop, k=200, fitness_attr="fitness_value", minimize=True)
    avg = np.mean([s.fitness_value for s in selected])
    assert avg < 50.0


def test_roulette_uniform_when_all_same():
    pop = make_pop([1.0, 1.0, 1.0, 1.0])
    sel = RouletteSelection()
    selected = sel.select(pop, k=100, fitness_attr="fitness_value", minimize=True)
    assert len(selected) == 100


def test_rank_returns_k():
    pop = make_pop([0.5, 0.3, 0.8, 0.1, 0.6])
    sel = RankSelection(pressure=1.5)
    selected = sel.select(pop, k=3, fitness_attr="fitness_value", minimize=True)
    assert len(selected) == 3


def test_rank_tends_to_select_best():
    pop = make_pop([float(i) for i in range(100)])
    sel = RankSelection(pressure=2.0)
    selected = sel.select(pop, k=200, fitness_attr="fitness_value", minimize=True)
    avg = np.mean([s.fitness_value for s in selected])
    assert avg < 50.0


def test_rank_probabilities_sum_to_one():
    pop = make_pop([0.1, 0.5, 0.9])
    sel = RankSelection(pressure=1.5)
    sel.select(pop, k=10, fitness_attr="fitness_value", minimize=True)


def test_selection_pluggable_in_population():
    import numpy as np
    from symgene import PrimitiveSet, Population
    from symgene.primitives import STANDARD
    pset = PrimitiveSet(n_inputs=2)
    pset.add_from_catalog(STANDARD)
    sel = RouletteSelection()
    pop = Population("p", pset, n_genes=2, pop_size=10, n_genes_max=4, selection=sel)
    pop.initialize(seed=0)
    pop.evaluate(np.random.rand(20, 2), np.random.rand(20))


from symgene.selection.lexicase import LexicaseSelection


def _make_pop_with_case_errors(n_inds=10, n_cases=20):
    rng = np.random.default_rng(7)

    class IndWithErrors:
        def __init__(self, errors, fitness_value):
            self._case_errors = errors
            self.fitness_value = fitness_value

    return [
        IndWithErrors(rng.uniform(0, 1, n_cases), rng.uniform(0, 1))
        for _ in range(n_inds)
    ]


def test_lexicase_returns_k():
    pop = _make_pop_with_case_errors(10, 20)
    sel = LexicaseSelection()
    selected = sel.select(pop, k=5)
    assert len(selected) == 5


def test_lexicase_all_selected_from_pool():
    pop = _make_pop_with_case_errors(8, 15)
    sel = LexicaseSelection()
    selected = sel.select(pop, k=20)
    for s in selected:
        assert s in pop


def test_lexicase_epsilon_selects_broader_pool():
    pop = _make_pop_with_case_errors(20, 50)
    sel_strict = LexicaseSelection(epsilon=0.0)
    sel_eps = LexicaseSelection(epsilon=0.1)
    assert len(sel_strict.select(pop, k=5)) == 5
    assert len(sel_eps.select(pop, k=5)) == 5


def test_lexicase_fallback_without_case_errors():
    pop = make_pop([0.5, 0.3, 0.8, 0.1])  # FakeInd without _case_errors
    sel = LexicaseSelection()
    selected = sel.select(pop, k=2, fitness_attr="fitness_value")
    assert len(selected) == 2


def test_population_stores_case_errors():
    import numpy as np
    from symgene import PrimitiveSet, Population
    from symgene.primitives import STANDARD
    pset = PrimitiveSet(n_inputs=2)
    pset.add_from_catalog(STANDARD)
    pop = Population("p", pset, n_genes=2, pop_size=8, n_genes_max=4)
    pop.initialize(seed=0)
    X = np.random.rand(30, 2)
    y = np.random.rand(30)
    pop.evaluate(X, y)
    found = False
    for ind in pop._population:
        if hasattr(ind, "_case_errors") and ind._case_errors is not None:
            assert len(ind._case_errors) == 30
            found = True
    assert found

from symgene.selection.lexicase import LexicaseSelection


def _make_pop_with_case_errors(n_inds=10, n_cases=20):
    rng = np.random.default_rng(7)

    class IndWithErrors:
        def __init__(self, errors, fitness_value):
            self._case_errors = errors
            self.fitness_value = fitness_value

    return [
        IndWithErrors(rng.uniform(0, 1, n_cases), rng.uniform(0, 1))
        for _ in range(n_inds)
    ]


def test_lexicase_returns_k():
    pop = _make_pop_with_case_errors(10, 20)
    sel = LexicaseSelection()
    selected = sel.select(pop, k=5)
    assert len(selected) == 5


def test_lexicase_all_selected_from_pool():
    pop = _make_pop_with_case_errors(8, 15)
    sel = LexicaseSelection()
    selected = sel.select(pop, k=20)
    for s in selected:
        assert s in pop


def test_lexicase_epsilon_no_error():
    pop = _make_pop_with_case_errors(20, 50)
    sel_strict = LexicaseSelection(epsilon=0.0)
    sel_eps = LexicaseSelection(epsilon=0.1)
    assert len(sel_strict.select(pop, k=5)) == 5
    assert len(sel_eps.select(pop, k=5)) == 5


def test_lexicase_fallback_without_case_errors():
    pop = make_pop([0.5, 0.3, 0.8, 0.1])
    sel = LexicaseSelection()
    selected = sel.select(pop, k=2, fitness_attr="fitness_value")
    assert len(selected) == 2


def test_population_stores_case_errors():
    import numpy as np
    from symgene import PrimitiveSet, Population
    from symgene.primitives import STANDARD
    pset = PrimitiveSet(n_inputs=2)
    pset.add_from_catalog(STANDARD)
    pop = Population("p", pset, n_genes=2, pop_size=8, n_genes_max=4)
    pop.initialize(seed=0)
    X = np.random.rand(30, 2)
    y = np.random.rand(30)
    pop.evaluate(X, y)
    found = False
    for ind in pop._population:
        if hasattr(ind, "_case_errors") and ind._case_errors is not None:
            assert len(ind._case_errors) == 30
            found = True
    assert found
