from unittest.mock import MagicMock
from symgene.advisor.monitor import EvolutionMonitor, MonitorSnapshot


def _make_individual(fitness_val: float):
    ind = MagicMock()
    ind.fitness.values = (fitness_val,)
    ind.fitness.valid = True
    ind.__str__ = MagicMock(return_value=str(fitness_val))
    genes = [MagicMock() for _ in range(3)]
    for g in genes:
        g.__len__ = MagicMock(return_value=5)
    ind.__iter__ = MagicMock(return_value=iter(genes))
    ind.__len__ = MagicMock(return_value=3)
    return ind


def _make_pop(name: str, fitness_values: list):
    pop = MagicMock()
    pop.name = name
    pop._population = [_make_individual(v) for v in fitness_values]
    pop.pset.primitives = [("add_fn", 2, "add", None), ("sin_fn", 1, "sin", None)]
    pop.pset.feature_names = ["x1", "x2"]
    best_ind = pop._population[0]
    best_ind.fitness.values = (min(fitness_values),)
    pop.best = best_ind
    return pop


def test_snapshot_is_dataclass():
    snap = MonitorSnapshot(
        gen=10, pop_name="p", stagnation_counter=5, diversity_index=0.8,
        complexity_trend=0.02, fitness_variance=0.001, best_fitness=0.05,
        best_expression_str="x1", fitness_history_last_30=[0.05],
        available_primitives=["add", "sin"], overfit_gap=None,
    )
    assert snap.gen == 10
    assert snap.stagnation_counter == 5
    assert snap.overfit_gap is None


def test_monitor_stagnation_counter_starts_at_zero():
    monitor = EvolutionMonitor()
    pop = _make_pop("p", [0.05, 0.06, 0.07])
    snap = monitor.update(gen=0, pop=pop, history=[], val_fitness=None)
    assert snap.stagnation_counter == 0


def test_monitor_stagnation_increments_without_improvement():
    monitor = EvolutionMonitor()
    pop = _make_pop("p", [0.05, 0.06, 0.07])
    monitor.update(gen=0, pop=pop, history=[], val_fitness=None)
    snap = monitor.update(gen=1, pop=pop, history=[], val_fitness=None)
    assert snap.stagnation_counter == 1


def test_monitor_stagnation_resets_on_improvement():
    monitor = EvolutionMonitor()
    pop = _make_pop("p", [0.05, 0.06])
    monitor.update(gen=0, pop=pop, history=[], val_fitness=None)
    monitor.update(gen=1, pop=pop, history=[], val_fitness=None)
    pop2 = _make_pop("p", [0.03, 0.04])
    snap = monitor.update(gen=2, pop=pop2, history=[], val_fitness=None)
    assert snap.stagnation_counter == 0


def test_monitor_snapshot_has_correct_pop_name_and_gen():
    monitor = EvolutionMonitor()
    pop = _make_pop("myPop", [0.1])
    snap = monitor.update(gen=42, pop=pop, history=[], val_fitness=None)
    assert snap.pop_name == "myPop"
    assert snap.gen == 42


def test_monitor_overfit_gap_is_none_without_val():
    monitor = EvolutionMonitor()
    pop = _make_pop("p", [0.05])
    snap = monitor.update(gen=0, pop=pop, history=[], val_fitness=None)
    assert snap.overfit_gap is None


def test_monitor_overfit_gap_computed_with_val():
    monitor = EvolutionMonitor()
    pop = _make_pop("p", [0.05])
    history = [{"gen": 0, "train_mse": 0.05}]
    snap = monitor.update(gen=0, pop=pop, history=history, val_fitness=0.08)
    assert snap.overfit_gap == 0.08


def test_monitor_primitives_extracted_from_pset():
    monitor = EvolutionMonitor()
    pop = _make_pop("p", [0.1])
    snap = monitor.update(gen=0, pop=pop, history=[], val_fitness=None)
    assert "add" in snap.available_primitives
    assert "sin" in snap.available_primitives
