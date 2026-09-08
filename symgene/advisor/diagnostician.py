from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Any
from symgene.advisor.monitor import MonitorSnapshot

VALID_ACTIONS = {
    "genetic_rescue",
    "adjust_mutation_rate",
    "force_migration",
    "restart_worst",
    "suggest_early_stop",
}

_SYSTEM_PROMPT = """You are SymGene Advisor, an expert in evolutionary symbolic regression.
Analyze the population state snapshot and recommend one targeted intervention.
Respond ONLY with valid JSON matching this exact schema:
{
  "root_cause": "<premature_convergence|bloat|overfitting|diversity_collapse|instability>",
  "severity": "<low|medium|high>",
  "confidence": <float 0.0-1.0>,
  "recommended_action": "<genetic_rescue|adjust_mutation_rate|force_migration|restart_worst|suggest_early_stop>",
  "action_params": {<see below>},
  "advisor_message": "<2-3 sentence description: what was detected, why, what action is taken>"
}

action_params per action:
- genetic_rescue: {"rescue_fraction": <0.1-0.3>, "level": "<gene|individual>"}
- adjust_mutation_rate: {"new_mutpb": <0.1-0.5>}
- force_migration: {}
- restart_worst: {"fraction": <0.1-0.3>}
- suggest_early_stop: {}"""


def _build_prompt(snapshot: MonitorSnapshot) -> str:
    return (
        f"Population '{snapshot.pop_name}' at generation {snapshot.gen}:\n\n"
        f"Stagnation counter: {snapshot.stagnation_counter} generations\n"
        f"Diversity index: {snapshot.diversity_index:.3f} (1.0 = fully diverse)\n"
        f"Complexity trend: {snapshot.complexity_trend:+.3f} (positive = growing nodes)\n"
        f"Fitness variance: {snapshot.fitness_variance:.6f}\n"
        f"Best fitness: {snapshot.best_fitness:.6f}\n"
        f"Best expression: {snapshot.best_expression_str[:200]}\n"
        f"Overfit gap (val-train): {snapshot.overfit_gap}\n"
        f"Recent fitness (last {len(snapshot.fitness_history_last_30)} gens): "
        f"{[round(v, 6) for v in snapshot.fitness_history_last_30[-10:]]}\n"
        f"Available primitives: {snapshot.available_primitives}\n\n"
        "Diagnose the most critical issue and recommend one intervention."
    )


@dataclass
class Diagnosis:
    root_cause: str
    severity: str
    confidence: float
    recommended_action: str
    action_params: dict[str, Any]
    advisor_message: str


def _parse_response(raw: str) -> Diagnosis:
    text = raw.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    data = json.loads(text)
    action = data.get("recommended_action", "genetic_rescue")
    if action not in VALID_ACTIONS:
        action = "genetic_rescue"

    return Diagnosis(
        root_cause=data.get("root_cause", "unknown"),
        severity=data.get("severity", "medium"),
        confidence=float(data.get("confidence", 0.5)),
        recommended_action=action,
        action_params=data.get("action_params", {}),
        advisor_message=str(data.get("advisor_message", "")),
    )


class Diagnostician:
    def __init__(self, client: Any) -> None:
        self.client = client

    def diagnose(self, snapshot: MonitorSnapshot) -> Diagnosis:
        prompt = _build_prompt(snapshot)
        raw = self.client.complete(prompt, system=_SYSTEM_PROMPT)
        return _parse_response(raw)
