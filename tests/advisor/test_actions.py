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


def test_unknown_action_fallback():
    executor = ActionExecutor(client=None)
    pop = _make_pop()
    result = executor.execute(_diag("unknown_action_xyz"), pop, gen=5)
    assert result["action"] == "unknown_action_xyz"
    assert result["n_affected"] == 0


def test_adjust_parameters_cxpb_and_cxpb_low():
    executor = ActionExecutor(client=None)
    pop = _make_pop()
    pop.cxpb = 0.8
    pop.cxpb_low = 0.4
    result = executor.execute(
        _diag("adjust_parameters", {"cxpb": 0.95, "cxpb_low": 0.6}), pop, gen=5
    )
    assert pop.cxpb == 0.95
    assert pop.cxpb_low == 0.6
    assert "cxpb" in result["changes"]
    assert "cxpb_low" in result["changes"]


def test_adjust_parameters_elite_ratio():
    executor = ActionExecutor(client=None)
    pop = _make_pop()
    pop.elite_ratio = 0.05
    result = executor.execute(
        _diag("adjust_parameters", {"elite_ratio": 0.08}), pop, gen=5
    )
    assert pop.elite_ratio == pytest.approx(0.08)


def test_suggest_primitive_missing_name():
    executor = ActionExecutor(client=None)
    pop = _make_pop()
    result = executor.execute(
        _diag("suggest_primitive", {"lambda_str": "lambda x: x**2"}), pop, gen=5
    )
    assert result["action"] == "suggest_primitive"
    assert result["n_affected"] == 0
    assert "missing" in result["reason"]


def test_suggest_primitive_bad_lambda():
    executor = ActionExecutor(client=None)
    pop = _make_pop()
    result = executor.execute(
        _diag("suggest_primitive", {"name": "bad_fn", "lambda_str": "invalid"}),
        pop, gen=5
    )
    assert result["action"] == "suggest_primitive"
    assert result["n_affected"] == 0


from symgene.advisor.actions import (
    _deap_fn_names, _deap_var_names, _deap_ephemeral_names, _build_top_context,
    _fix_ephemerals,
)
from symgene.primitive_set import PrimitiveSet
from symgene.population import Population


def _real_pop():
    pset = PrimitiveSet(n_inputs=2, feature_names=["x1", "x2"])
    pset.add_from_catalog(["add", "sin"])
    pop = Population(name="p", pset=pset, n_genes=2, pop_size=10)
    pop.initialize(seed=0)
    return pop


def test_deap_fn_names_nonempty():
    pop = _real_pop()
    names = _deap_fn_names(pop)
    assert "add" in names


def test_deap_var_names_nonempty():
    pop = _real_pop()
    names = _deap_var_names(pop)
    assert "x1" in names or "ARG0" in names  # DEAP may rename to ARG0/ARG1


def test_deap_ephemeral_names():
    pop = _real_pop()
    names = _deap_ephemeral_names(pop)
    assert isinstance(names, list)


def test_build_top_context_with_evaluated_pop():
    import numpy as np
    from symgene.fitness import FitnessEvaluator
    from symgene.metrics.regression import mse
    pset = PrimitiveSet(n_inputs=2, feature_names=["x1", "x2"])
    pset.add_from_catalog(["add", "sin"])
    pop = Population(name="p", pset=pset, n_genes=2, pop_size=10,
                     fitness=FitnessEvaluator(metric=mse), combiner="ridge")
    pop.initialize(seed=0)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 2))
    y = X[:, 0] + X[:, 1]
    pop.evaluate(X, y)
    ctx = _build_top_context(pop, k=2)
    assert isinstance(ctx, str)
    assert len(ctx) > 0


def test_suggest_primitive_valid_with_real_pop():
    """A valid lambda with a real initialized population registers the primitive."""
    pset = PrimitiveSet(n_inputs=2, feature_names=["x1", "x2"])
    pset.add_from_catalog(["add", "sin"])
    pop = Population(name="p", pset=pset, n_genes=2, pop_size=5)
    pop.initialize(seed=0)
    # Patch DEAP's addPrimitive to avoid DEAP internal state issues
    pop._deap_pset.addPrimitive = MagicMock()
    executor = ActionExecutor(client=None)
    result = executor.execute(
        _diag("suggest_primitive", {
            "name": "mysquare",
            "lambda_str": "lambda x: x * x",
            "arity": 1,
            "description": "square function",
        }),
        pop, gen=5,
    )
    assert result["action"] == "suggest_primitive"
    assert result["n_affected"] == 1
    assert result["name"] == "mysquare"


def test_suggest_primitive_valid_arity2_no_squash():
    """arity=2, no squash: goes through the no-squash branch for fn_final."""
    pset = PrimitiveSet(n_inputs=2, feature_names=["x1", "x2"])
    pset.add_from_catalog(["add"])
    pset.disable_squash()
    pop = Population(name="p", pset=pset, n_genes=2, pop_size=5)
    pop.initialize(seed=0)
    pop._deap_pset.addPrimitive = MagicMock()
    executor = ActionExecutor(client=None)
    result = executor.execute(
        _diag("suggest_primitive", {
            "name": "mymul",
            "lambda_str": "lambda x, y: x * y",
            "arity": 2,
            "description": "multiply",
        }),
        pop, gen=5,
    )
    assert result["action"] == "suggest_primitive"
    assert result["n_affected"] == 1


def test_adjust_parameters_tournament_size():
    executor = ActionExecutor(client=None)
    pop = _make_pop()
    pop.selection = MagicMock()
    pop.selection.size = 5
    result = executor.execute(
        _diag("adjust_parameters", {"tournament_size": 3}), pop, gen=5
    )
    assert pop.selection.size == 3
    assert "tournament_size" in result["changes"]


import pytest
