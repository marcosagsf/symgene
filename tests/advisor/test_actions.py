from unittest.mock import MagicMock
from symgene.advisor.actions import ActionExecutor
from symgene.advisor.diagnostician import Diagnosis


def _diag(action: str, params: dict | None = None) -> Diagnosis:
    return Diagnosis(
        root_cause="test", severity="medium", confidence=0.8,
        recommended_action=action,
        action_params=params or {},
        advisor_message="test",
    )


def _make_ind():
    ind = MagicMock()
    ind.fitness.valid = True
    ind.fitness.values = (0.05,)
    ind.__len__ = MagicMock(return_value=3)
    ind.__iter__ = MagicMock(return_value=iter([MagicMock(), MagicMock(), MagicMock()]))
    return ind


def _make_pop(mutpb: float = 0.2, n_inds: int = 20):
    pop = MagicMock()
    pop.name = "p"
    pop.pop_size = n_inds
    pop.mutpb = mutpb
    pop._population = [_make_ind() for _ in range(n_inds)]
    pop.best = pop._population[0]
    pop.pset.primitives = [("add", 2, "add", None)]
    pop.pset.feature_names = ["x1"]
    pop._toolbox = MagicMock()
    pop._toolbox.individual.return_value = MagicMock()
    return pop


def test_adjust_mutation_rate_updates_pop_mutpb():
    executor = ActionExecutor(client=None)
    pop = _make_pop(mutpb=0.2)
    result = executor.execute(_diag("adjust_mutation_rate", {"new_mutpb": 0.35}), pop, gen=10)
    assert pop.mutpb == 0.35
    assert result["action"] == "adjust_mutation_rate"
    assert result["changes"]["mutpb"]["from"] == 0.2
    assert result["changes"]["mutpb"]["to"] == 0.35


def test_adjust_mutation_rate_clips_to_max():
    executor = ActionExecutor(client=None)
    pop = _make_pop()
    executor.execute(_diag("adjust_mutation_rate", {"new_mutpb": 0.99}), pop, gen=10)
    assert pop.mutpb <= 0.5


def test_adjust_mutation_rate_clips_to_min():
    executor = ActionExecutor(client=None)
    pop = _make_pop()
    executor.execute(_diag("adjust_mutation_rate", {"new_mutpb": 0.001}), pop, gen=10)
    assert pop.mutpb >= 0.05


def test_genetic_rescue_without_client_returns_zero_affected():
    executor = ActionExecutor(client=None)
    pop = _make_pop()
    result = executor.execute(_diag("genetic_rescue", {"rescue_fraction": 0.15}), pop, gen=10)
    assert result["action"] == "genetic_rescue"
    assert result["n_affected"] == 0
    assert "reason" in result


def test_restart_worst_reinitializes_correct_count():
    executor = ActionExecutor(client=None)
    pop = _make_pop(n_inds=20)
    result = executor.execute(_diag("restart_worst", {"fraction": 0.2}), pop, gen=10)
    assert result["action"] == "restart_worst"
    assert result["n_affected"] == 4
    # 4 individuals × 3 genes each = 12 gene() calls
    assert pop._toolbox.gene.call_count == 4 * 3


def test_suggest_early_stop_returns_action_name():
    executor = ActionExecutor(client=None)
    pop = _make_pop()
    result = executor.execute(_diag("suggest_early_stop"), pop, gen=10)
    assert result["action"] == "suggest_early_stop"


def test_force_migration_single_pop_returns_zero_affected():
    executor = ActionExecutor(client=None)
    pop = _make_pop()
    result = executor.execute(_diag("force_migration"), pop, gen=10)
    assert result["action"] == "force_migration"
    assert result["n_affected"] == 0
