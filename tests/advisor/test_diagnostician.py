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
