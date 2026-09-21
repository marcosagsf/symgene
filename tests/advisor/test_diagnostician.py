import json
from unittest.mock import MagicMock
from symgene.advisor.diagnostician import (
    Diagnostician, Diagnosis, _parse_response, VALID_ACTIONS
)
from symgene.advisor.monitor import MonitorSnapshot


def _snap() -> MonitorSnapshot:
    return MonitorSnapshot(
        gen=47, pop_name="p", stagnation_counter=23, diversity_index=0.09,
        complexity_trend=0.08, fitness_variance=0.001, best_fitness=0.024,
        best_expression_str="add(x1, sin(x2))",
        fitness_history_last_30=[0.024] * 10,
        available_primitives=["add", "sin", "mul"],
        overfit_gap=None,
    )


def _valid_json(action: str = "genetic_rescue") -> str:
    return json.dumps({
        "root_cause": "premature_convergence",
        "severity": "high",
        "confidence": 0.88,
        "recommended_action": action,
        "action_params": {"rescue_fraction": 0.15, "level": "individual"},
        "advisor_message": "Stagnation detected. Injecting new individuals.",
    })


def test_parse_valid_json():
    diag = _parse_response(_valid_json())
    assert diag.root_cause == "premature_convergence"
    assert diag.recommended_action == "genetic_rescue"
    assert diag.confidence == 0.88
    assert isinstance(diag.action_params, dict)
    assert diag.advisor_message != ""


def test_parse_json_with_markdown_fence():
    raw = f"```json\n{_valid_json()}\n```"
    diag = _parse_response(raw)
    assert diag.recommended_action == "genetic_rescue"


def test_parse_json_with_generic_fence():
    raw = f"```\n{_valid_json()}\n```"
    diag = _parse_response(raw)
    assert diag.recommended_action == "genetic_rescue"


def test_invalid_action_falls_back_to_genetic_rescue():
    data = json.loads(_valid_json())
    data["recommended_action"] = "destroy_everything"
    diag = _parse_response(json.dumps(data))
    assert diag.recommended_action == "genetic_rescue"


def test_valid_actions_set_is_complete():
    assert "genetic_rescue" in VALID_ACTIONS
    assert "adjust_mutation_rate" in VALID_ACTIONS
    assert "force_migration" in VALID_ACTIONS
    assert "restart_worst" in VALID_ACTIONS
    assert "suggest_early_stop" in VALID_ACTIONS


def test_diagnostician_calls_client_complete():
    client = MagicMock()
    client.complete.return_value = _valid_json()
    engine = Diagnostician(client)
    result = engine.diagnose(_snap())
    assert client.complete.called
    assert isinstance(result, Diagnosis)


def test_diagnostician_passes_system_prompt():
    client = MagicMock()
    client.complete.return_value = _valid_json()
    engine = Diagnostician(client)
    engine.diagnose(_snap())
    _, kwargs = client.complete.call_args
    assert "system" in kwargs
    assert "JSON" in kwargs["system"]


# --- private formatter helpers ---

from symgene.advisor.diagnostician import (
    _fmt_top_individuals, _fmt_primitive_usage,
    _fmt_intervention_history, _repair_truncated_json,
)

def test_fmt_top_individuals_nonempty():
    top = [
        {"fitness": 0.012, "n_genes": 2,
         "genes": [{"len": 5, "expr": "add(x1, x2)", "coef": 0.8}],
         "bias": 0.1},
    ]
    out = _fmt_top_individuals(top)
    assert "0.012000" in out
    assert "add(x1, x2)" in out

def test_fmt_top_individuals_empty():
    assert _fmt_top_individuals([]) == "  (none available)"

def test_fmt_primitive_usage_nonempty():
    usage = {"add": 10, "sin": 3}
    available = ["add", "sin", "mul"]
    out = _fmt_primitive_usage(usage, available)
    assert "add(10)" in out
    assert "mul" in out  # in NEVER used

def test_fmt_primitive_usage_empty():
    assert _fmt_primitive_usage({}, ["add"]) == "  (no data)"

def test_fmt_intervention_history_nonempty():
    history = [
        {"action": "genetic_rescue", "gen": 10, "n_affected": 3,
         "status": "ok", "root_cause": "stagnation",
         "changes": {"mutpb": {"from": 0.2, "to": 0.3}}},
    ]
    out = _fmt_intervention_history(history)
    assert "genetic_rescue" in out
    assert "stagnation" in out

def test_fmt_intervention_history_empty():
    assert _fmt_intervention_history([]) == "  (no prior interventions)"

def test_repair_truncated_json_closes_object():
    truncated = '{"key": "val'
    repaired = _repair_truncated_json(truncated)
    import json
    data = json.loads(repaired)
    assert "key" in data

def test_repair_truncated_json_closes_array():
    truncated = '["a", "b'
    repaired = _repair_truncated_json(truncated)
    import json
    data = json.loads(repaired)
    assert isinstance(data, list)

def test_parse_response_truncated_json_recovery():
    truncated = '{"recommended_action": "genetic_rescue", "root_cause": "stagnation", "severity": "high", "confidence": 0.9, "action_params": {}, "advisor_message": "test'
    diag = _parse_response(truncated)
    assert diag.recommended_action == "genetic_rescue"

def test_parse_response_regex_fallback():
    bad_json = 'blah blah "recommended_action": "restart_worst" blah'
    diag = _parse_response(bad_json)
    assert diag.recommended_action == "restart_worst"

def test_build_prompt_with_val_r2():
    from symgene.advisor.diagnostician import _build_prompt
    snap = MonitorSnapshot(
        gen=50, pop_name="p", stagnation_counter=10, diversity_index=0.5,
        complexity_trend=0.02, fitness_variance=0.01, best_fitness=0.01,
        best_expression_str="add(x1, x2)",
        fitness_history_last_30=[0.01] * 10,
        available_primitives=["add", "sin"],
        overfit_gap=0.05,
        val_r2=0.95,
        val_r2_history_last_10=[0.90, 0.91, 0.92, 0.93, 0.94, 0.95,
                                 0.95, 0.95, 0.95, 0.95],
        current_params={"mutpb": 0.2, "cxpb": 0.9},
    )
    prompt = _build_prompt(snap)
    assert "val_r2" in prompt.lower() or "r2" in prompt.lower()

def test_build_prompt_degrading_trend():
    from symgene.advisor.diagnostician import _build_prompt
    snap = MonitorSnapshot(
        gen=20, pop_name="p", stagnation_counter=5, diversity_index=0.5,
        complexity_trend=0.0, fitness_variance=0.1, best_fitness=0.1,
        best_expression_str="x1",
        fitness_history_last_30=[0.1] * 5,
        available_primitives=["add"],
        overfit_gap=None,
        val_r2=0.5,
        val_r2_history_last_10=[0.9, 0.85, 0.80, 0.75, 0.70,
                                 0.65, 0.60, 0.55, 0.52, 0.50],
        current_params={},
    )
    prompt = _build_prompt(snap)
    assert "DEGRADING" in prompt

def test_build_prompt_improving_trend():
    from symgene.advisor.diagnostician import _build_prompt
    snap = MonitorSnapshot(
        gen=20, pop_name="p", stagnation_counter=0, diversity_index=0.8,
        complexity_trend=0.0, fitness_variance=0.1, best_fitness=0.05,
        best_expression_str="mul(x1, x2)",
        fitness_history_last_30=[0.1] * 5,
        available_primitives=["mul"],
        overfit_gap=None,
        val_r2=0.7,
        val_r2_history_last_10=[0.5, 0.6, 0.65, 0.67, 0.68, 0.69,
                                 0.70, 0.71, 0.72, 0.73],
        current_params={},
    )
    prompt = _build_prompt(snap)
    assert "IMPROVING" in prompt
