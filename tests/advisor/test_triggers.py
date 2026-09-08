from symgene.advisor.monitor import MonitorSnapshot
from symgene.advisor.triggers import StagnationTrigger, DiversityTrigger, BloatTrigger


def _snap(**kwargs) -> MonitorSnapshot:
    defaults = dict(
        gen=10, pop_name="p", stagnation_counter=0, diversity_index=0.8,
        complexity_trend=0.0, fitness_variance=0.001, best_fitness=0.05,
        best_expression_str="x1", fitness_history_last_30=[0.05],
        available_primitives=["add", "sin"], overfit_gap=None,
    )
    defaults.update(kwargs)
    return MonitorSnapshot(**defaults)


def test_stagnation_trigger_fires_at_patience():
    t = StagnationTrigger(patience=10)
    assert t.evaluate(_snap(stagnation_counter=10)) is True


def test_stagnation_trigger_does_not_fire_below_patience():
    t = StagnationTrigger(patience=10)
    assert t.evaluate(_snap(stagnation_counter=9)) is False


def test_diversity_trigger_fires_below_threshold():
    t = DiversityTrigger(threshold=0.15)
    assert t.evaluate(_snap(diversity_index=0.10)) is True


def test_diversity_trigger_does_not_fire_above_threshold():
    t = DiversityTrigger(threshold=0.15)
    assert t.evaluate(_snap(diversity_index=0.20)) is False


def test_bloat_trigger_fires_above_growth():
    t = BloatTrigger(complexity_growth=0.10)
    assert t.evaluate(_snap(complexity_trend=0.15)) is True


def test_bloat_trigger_does_not_fire_below_growth():
    t = BloatTrigger(complexity_growth=0.10)
    assert t.evaluate(_snap(complexity_trend=0.05)) is False


def test_trigger_name_is_class_name():
    assert StagnationTrigger().name == "StagnationTrigger"
    assert DiversityTrigger().name == "DiversityTrigger"
    assert BloatTrigger().name == "BloatTrigger"
