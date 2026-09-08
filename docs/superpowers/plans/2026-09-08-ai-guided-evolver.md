# AIGuidedEvolver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `symgene/advisor/` — an optional LLM-guided evolution module that monitors training signals, triggers targeted LLM interventions, and prints a clear advisor panel inline with training output, without touching any existing symgene code.

**Architecture:** `AIGuidedEvolver` wraps `SymGeneEvolver` by composition, driving the generation loop itself and inserting advisor hooks after each generation. A zero-cost `EvolutionMonitor` tracks signals every generation; when a `BaseTrigger` fires, a two-stage LLM call (diagnosis → action) intervenes on the population. All LLM failures are silent — the run always completes.

**Tech Stack:** Python 3.10+, DEAP, NumPy, existing `symgene.llm.LLMClient`, existing `symgene.metrics.diversity`, `dataclasses`, `json`, `textwrap`

---

## Task 1: MonitorSnapshot + EvolutionMonitor

**Files:**
- Create: `symgene/advisor/monitor.py`
- Create: `tests/advisor/__init__.py` (empty)
- Create: `tests/advisor/test_monitor.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/advisor/test_monitor.py
from unittest.mock import MagicMock
from symgene.advisor.monitor import EvolutionMonitor, MonitorSnapshot


def _make_individual(fitness_val: float):
    ind = MagicMock()
    ind.fitness.values = (fitness_val,)
    ind.fitness.valid = True
    # str used by genotypic_diversity — unique per value
    ind.__str__ = MagicMock(return_value=str(fitness_val))
    # genes: list of 3 mock genes, each with len=5 nodes
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
    pop.pset.primitives = [("add", 2, "add", None), ("sin", 1, "sin", None)]
    pop.pset.feature_names = ["x1", "x2"]
    pop.best = pop._population[0]
    pop.best.fitness.values = (min(fitness_values),)
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
    # improve
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
```

- [ ] **Step 2: Run tests to verify they fail**

```
cd C:\Users\Marcos Filho\PycharmProjects\symgene
python -m pytest tests/advisor/test_monitor.py -v
```
Expected: `ModuleNotFoundError: No module named 'symgene.advisor'`

- [ ] **Step 3: Create empty `tests/advisor/__init__.py`**

Create `tests/advisor/__init__.py` — empty file.

- [ ] **Step 4: Implement `symgene/advisor/monitor.py`**

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from symgene.metrics.diversity import genotypic_diversity


@dataclass
class MonitorSnapshot:
    gen: int
    pop_name: str
    stagnation_counter: int
    diversity_index: float
    complexity_trend: float
    fitness_variance: float
    best_fitness: float
    best_expression_str: str
    fitness_history_last_30: list[float]
    available_primitives: list[str]
    overfit_gap: float | None


class EvolutionMonitor:
    def __init__(self) -> None:
        self._stagnation_counters: dict[str, int] = {}
        self._best_fitness_seen: dict[str, float] = {}
        self._complexity_history: dict[str, list[int]] = {}

    def update(
        self,
        gen: int,
        pop: Any,
        history: list[dict[str, Any]],
        val_fitness: float | None = None,
    ) -> MonitorSnapshot:
        name = pop.name

        if name not in self._best_fitness_seen:
            self._best_fitness_seen[name] = float("inf")
            self._stagnation_counters[name] = 0

        current_best = (
            pop.best.fitness.values[0] if pop.best is not None else float("inf")
        )
        if current_best < self._best_fitness_seen[name] - 1e-8:
            self._best_fitness_seen[name] = current_best
            self._stagnation_counters[name] = 0
        else:
            self._stagnation_counters[name] += 1

        div = genotypic_diversity(pop._population) if pop._population else 0.0

        if name not in self._complexity_history:
            self._complexity_history[name] = []
        current_nodes = int(np.mean([
            sum(len(g) for g in ind)
            for ind in pop._population
        ])) if pop._population else 0
        self._complexity_history[name].append(current_nodes)
        hist_c = self._complexity_history[name]
        if len(hist_c) >= 10 and self._stagnation_counters[name] >= 5:
            trend = (hist_c[-1] - hist_c[-10]) / max(hist_c[-10], 1)
        else:
            trend = 0.0

        valid_fits = [
            ind.fitness.values[0]
            for ind in pop._population
            if hasattr(ind, "fitness") and ind.fitness.valid
        ]
        fitness_var = float(np.var(valid_fits)) if valid_fits else 0.0

        fit_history = [h.get("train_mse", float("inf")) for h in history[-30:]]
        primitives = [pname for _, _, pname, *_ in pop.pset.primitives]

        overfit = None
        if val_fitness is not None and history:
            last_train = history[-1].get("train_mse", float("inf"))
            overfit = float(val_fitness - last_train)

        best_str = ""
        if pop.best is not None:
            try:
                best_str = " | ".join(str(g) for g in pop.best)
            except Exception:
                best_str = str(pop.best)

        return MonitorSnapshot(
            gen=gen,
            pop_name=name,
            stagnation_counter=self._stagnation_counters[name],
            diversity_index=div,
            complexity_trend=trend,
            fitness_variance=fitness_var,
            best_fitness=current_best,
            best_expression_str=best_str,
            fitness_history_last_30=fit_history,
            available_primitives=primitives,
            overfit_gap=overfit,
        )
```

- [ ] **Step 5: Run tests to verify they pass**

```
python -m pytest tests/advisor/test_monitor.py -v
```
Expected: all 8 tests PASS

- [ ] **Step 6: Commit**

```
git add symgene/advisor/monitor.py tests/advisor/__init__.py tests/advisor/test_monitor.py
git commit -m "feat(advisor): add EvolutionMonitor and MonitorSnapshot"
```

---

## Task 2: Trigger Conditions

**Files:**
- Create: `symgene/advisor/triggers.py`
- Create: `tests/advisor/test_triggers.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/advisor/test_triggers.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

```
python -m pytest tests/advisor/test_triggers.py -v
```
Expected: `ImportError: cannot import name 'StagnationTrigger'`

- [ ] **Step 3: Implement `symgene/advisor/triggers.py`**

```python
from __future__ import annotations
from symgene.advisor.monitor import MonitorSnapshot


class BaseTrigger:
    def evaluate(self, snapshot: MonitorSnapshot) -> bool:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.__class__.__name__


class StagnationTrigger(BaseTrigger):
    def __init__(self, patience: int = 20) -> None:
        self.patience = patience

    def evaluate(self, snapshot: MonitorSnapshot) -> bool:
        return snapshot.stagnation_counter >= self.patience


class DiversityTrigger(BaseTrigger):
    def __init__(self, threshold: float = 0.15) -> None:
        self.threshold = threshold

    def evaluate(self, snapshot: MonitorSnapshot) -> bool:
        return snapshot.diversity_index < self.threshold


class BloatTrigger(BaseTrigger):
    def __init__(self, complexity_growth: float = 0.10) -> None:
        self.complexity_growth = complexity_growth

    def evaluate(self, snapshot: MonitorSnapshot) -> bool:
        return snapshot.complexity_trend > self.complexity_growth
```

- [ ] **Step 4: Run tests to verify they pass**

```
python -m pytest tests/advisor/test_triggers.py -v
```
Expected: all 7 tests PASS

- [ ] **Step 5: Commit**

```
git add symgene/advisor/triggers.py tests/advisor/test_triggers.py
git commit -m "feat(advisor): add StagnationTrigger, DiversityTrigger, BloatTrigger"
```

---

## Task 3: Diagnostician — Stage 1 LLM Call

**Files:**
- Create: `symgene/advisor/diagnostician.py`
- Create: `tests/advisor/test_diagnostician.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/advisor/test_diagnostician.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

```
python -m pytest tests/advisor/test_diagnostician.py -v
```
Expected: `ImportError: cannot import name 'Diagnostician'`

- [ ] **Step 3: Implement `symgene/advisor/diagnostician.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```
python -m pytest tests/advisor/test_diagnostician.py -v
```
Expected: all 8 tests PASS

- [ ] **Step 5: Commit**

```
git add symgene/advisor/diagnostician.py tests/advisor/test_diagnostician.py
git commit -m "feat(advisor): add Diagnostician with two-stage LLM diagnosis"
```

---

## Task 4: Action Executor — Stage 2

**Files:**
- Create: `symgene/advisor/actions.py`
- Create: `tests/advisor/test_actions.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/advisor/test_actions.py
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


def _make_pop(mutpb: float = 0.2, n_inds: int = 20):
    pop = MagicMock()
    pop.name = "p"
    pop.pop_size = n_inds
    pop.mutpb = mutpb
    ind = MagicMock()
    ind.fitness.valid = True
    ind.fitness.values = (0.05,)
    ind.__len__ = MagicMock(return_value=3)
    ind.__iter__ = MagicMock(return_value=iter([MagicMock(), MagicMock(), MagicMock()]))
    pop._population = [ind] * n_inds
    pop.best = ind
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
    assert result["old_mutpb"] == 0.2
    assert result["new_mutpb"] == 0.35


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
    assert result["n_affected"] == 4   # 20 * 0.2
    assert pop._toolbox.individual.call_count == 4


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
```

- [ ] **Step 2: Run tests to verify they fail**

```
python -m pytest tests/advisor/test_actions.py -v
```
Expected: `ImportError: cannot import name 'ActionExecutor'`

- [ ] **Step 3: Implement `symgene/advisor/actions.py`**

```python
from __future__ import annotations
import json
import random
from typing import Any
import deap.gp as gp
import deap.creator as creator

from symgene.advisor.diagnostician import Diagnosis

_RESCUE_SYSTEM = (
    "You are SymGene Advisor. Generate new symbolic expressions in DEAP prefix notation "
    "to inject into the genetic programming population. "
    "Respond ONLY with a JSON array of strings. "
    "Use only the provided primitives and variables. "
    'Example: ["add(x1, mul(sin(x1), x2))", "mul(log(add(x1, 1.0)), x2)"]'
)


class ActionExecutor:
    def __init__(self, client: Any | None = None) -> None:
        self.client = client

    def execute(self, diagnosis: Diagnosis, pop: Any, gen: int) -> dict[str, Any]:
        action = diagnosis.recommended_action
        if action == "genetic_rescue":
            return self._genetic_rescue(diagnosis, pop)
        if action == "adjust_mutation_rate":
            return self._adjust_mutation_rate(diagnosis, pop)
        if action == "force_migration":
            return {"action": "force_migration", "n_affected": 0}
        if action == "restart_worst":
            return self._restart_worst(diagnosis, pop)
        if action == "suggest_early_stop":
            return {"action": "suggest_early_stop", "n_affected": 0}
        return {"action": action, "n_affected": 0}

    def _genetic_rescue(self, diagnosis: Diagnosis, pop: Any) -> dict[str, Any]:
        if self.client is None:
            return {"action": "genetic_rescue", "n_affected": 0,
                    "reason": "no llm client"}

        params = diagnosis.action_params
        fraction = float(params.get("rescue_fraction", 0.15))
        level = params.get("level", "individual")
        n_rescue = max(1, int(fraction * pop.pop_size))

        best_str = ""
        if pop.best is not None:
            try:
                best_str = " | ".join(str(g) for g in pop.best)
            except Exception:
                pass
        primitives = [pname for _, _, pname, *_ in pop.pset.primitives]
        variables = pop.pset.feature_names

        prompt = (
            f"Generate {n_rescue} diverse symbolic expressions.\n"
            f"Available primitives: {primitives}\n"
            f"Variables: {variables}\n"
            f"Current best (generate structurally different forms): {best_str[:300]}\n"
            f"Level: {level}\n"
            f"Respond with a JSON array of exactly {n_rescue} DEAP prefix-notation strings."
        )

        rescued = 0
        for _attempt in range(3):
            try:
                raw = self.client.complete(prompt, system=_RESCUE_SYSTEM)
                text = raw.strip()
                if text.startswith("```"):
                    parts = text.split("```")
                    text = parts[1] if len(parts) > 1 else text
                    if text.startswith("json"):
                        text = text[4:]
                    text = text.strip()
                exprs = json.loads(text)
                if not isinstance(exprs, list):
                    continue

                worst_indices = sorted(
                    range(len(pop._population)),
                    key=lambda i: (
                        pop._population[i].fitness.values[0]
                        if pop._population[i].fitness.valid else 0.0
                    ),
                    reverse=True,
                )[:n_rescue]

                for idx, expr_str in zip(worst_indices, exprs):
                    try:
                        tree = gp.PrimitiveTree.from_string(expr_str, pop._deap_pset)
                        gene = creator.SGGene(tree)
                        if level == "gene":
                            replace_idx = random.randint(
                                0, len(pop._population[idx]) - 1
                            )
                            pop._population[idx][replace_idx] = gene
                        else:
                            n_genes = len(pop._population[idx])
                            new_ind = creator.SGIndividual(
                                creator.SGGene(
                                    gp.PrimitiveTree.from_string(expr_str, pop._deap_pset)
                                )
                                for _ in range(n_genes)
                            )
                            pop._population[idx] = new_ind
                        pop._population[idx].fitness.valid = False
                        rescued += 1
                    except Exception:
                        continue
                break
            except Exception:
                continue

        return {"action": "genetic_rescue", "n_affected": rescued}

    def _adjust_mutation_rate(self, diagnosis: Diagnosis, pop: Any) -> dict[str, Any]:
        new_mutpb = float(diagnosis.action_params.get("new_mutpb", 0.3))
        new_mutpb = max(0.05, min(0.5, new_mutpb))
        old = pop.mutpb
        pop.mutpb = new_mutpb
        return {"action": "adjust_mutation_rate", "n_affected": 1,
                "old_mutpb": old, "new_mutpb": new_mutpb}

    def _restart_worst(self, diagnosis: Diagnosis, pop: Any) -> dict[str, Any]:
        fraction = float(diagnosis.action_params.get("fraction", 0.2))
        n_restart = max(1, int(fraction * pop.pop_size))
        worst_indices = sorted(
            range(len(pop._population)),
            key=lambda i: (
                pop._population[i].fitness.values[0]
                if pop._population[i].fitness.valid else 0.0
            ),
            reverse=True,
        )[:n_restart]
        assert pop._toolbox is not None
        for idx in worst_indices:
            pop._population[idx] = pop._toolbox.individual()
            pop._population[idx].fitness.valid = False
        return {"action": "restart_worst", "n_affected": n_restart}
```

- [ ] **Step 4: Run tests to verify they pass**

```
python -m pytest tests/advisor/test_actions.py -v
```
Expected: all 7 tests PASS

- [ ] **Step 5: Commit**

```
git add symgene/advisor/actions.py tests/advisor/test_actions.py
git commit -m "feat(advisor): add ActionExecutor with deterministic and LLM-based actions"
```

---

## Task 5: AIGuidedEvolver

**Files:**
- Create: `symgene/advisor/guided_evolver.py`
- Create: `tests/advisor/test_guided_evolver.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/advisor/test_guided_evolver.py
import numpy as np
from symgene.population import Population
from symgene.primitive_set import PrimitiveSet
from symgene.advisor import AIGuidedEvolver, StagnationTrigger


def _make_pop(name: str = "target") -> Population:
    pset = PrimitiveSet(n_inputs=2, feature_names=["x1", "x2"])
    pset.add_from_catalog()
    return Population(name=name, pset=pset, n_genes=2, pop_size=10)


_X = np.random.default_rng(0).random((20, 2))
_y = {"target": _X[:, 0] + _X[:, 1]}


def test_aiguidedvolver_completes_without_llm():
    pop = _make_pop()
    guided = AIGuidedEvolver(
        populations=[pop], n_gen=5, llm_client=None, verbose=0,
    )
    result = guided.fit(_X, _y)
    assert "target" in result


def test_result_has_advisor_log_attribute():
    pop = _make_pop()
    guided = AIGuidedEvolver(
        populations=[pop], n_gen=5, llm_client=None, verbose=0,
    )
    result = guided.fit(_X, _y)
    assert hasattr(result, "advisor_log_")
    assert isinstance(result.advisor_log_, list)


def test_no_interventions_without_llm_client():
    pop = _make_pop()
    guided = AIGuidedEvolver(
        populations=[pop], n_gen=5, llm_client=None,
        triggers=[StagnationTrigger(patience=1)], verbose=0,
    )
    result = guided.fit(_X, _y)
    assert result.advisor_log_ == []


def test_result_predictions_have_correct_shape():
    pop = _make_pop()
    guided = AIGuidedEvolver(
        populations=[pop], n_gen=5, llm_client=None, verbose=0,
    )
    result = guided.fit(_X, _y)
    y_pred = result["target"].predict(_X)
    assert y_pred.shape == (20,)


def test_symgeneevolver_result_has_no_advisor_log():
    from symgene.evolver import SymGeneEvolver
    pop = _make_pop()
    evolver = SymGeneEvolver(populations=[pop], n_gen=5, verbose=0)
    result = evolver.fit(_X, _y)
    assert not hasattr(result, "advisor_log_")


def test_default_triggers_applied_when_none_provided():
    pop = _make_pop()
    guided = AIGuidedEvolver(
        populations=[pop], n_gen=3, llm_client=None, verbose=0,
    )
    assert len(guided.triggers) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

```
python -m pytest tests/advisor/test_guided_evolver.py -v
```
Expected: `ImportError: cannot import name 'AIGuidedEvolver'`

- [ ] **Step 3: Implement `symgene/advisor/guided_evolver.py`**

```python
from __future__ import annotations
import random
import textwrap
from typing import Any, TYPE_CHECKING

import numpy as np

from symgene.evolver import SymGeneEvolver
from symgene.population import Population
from symgene.metrics.regression import r2
from symgene.advisor.monitor import EvolutionMonitor
from symgene.advisor.triggers import (
    BaseTrigger, StagnationTrigger, DiversityTrigger, BloatTrigger,
)
from symgene.advisor.diagnostician import Diagnostician, Diagnosis
from symgene.advisor.actions import ActionExecutor

if TYPE_CHECKING:
    from symgene.results import SymGeneResult

_BORDER = "━" * 54


class AIGuidedEvolver:
    def __init__(
        self,
        populations: list[Population],
        n_gen: int = 200,
        llm_client: Any = None,
        triggers: list[BaseTrigger] | None = None,
        max_interventions: int = 8,
        cooldown: int = 15,
        seed: int | None = None,
        verbose: int = 1,
        **evolver_kwargs: Any,
    ) -> None:
        self._evolver = SymGeneEvolver(
            populations=populations,
            n_gen=n_gen,
            seed=seed,
            verbose=verbose,
            **evolver_kwargs,
        )
        self.llm_client = llm_client
        self.triggers: list[BaseTrigger] = triggers or [
            StagnationTrigger(patience=20),
            DiversityTrigger(threshold=0.15),
            BloatTrigger(complexity_growth=0.10),
        ]
        self.max_interventions = max_interventions
        self.cooldown = cooldown
        self.verbose = verbose

        self._monitor = EvolutionMonitor()
        self._diagnostician = Diagnostician(llm_client) if llm_client else None
        self._executor = ActionExecutor(llm_client)
        self._advisor_log: list[dict[str, Any]] = []

    def fit(
        self,
        X: np.ndarray,
        y: dict[str, np.ndarray],
        X_val: np.ndarray | None = None,
        y_val: dict[str, np.ndarray] | None = None,
    ) -> SymGeneResult:
        ev = self._evolver

        if ev.seed is not None:
            random.seed(ev.seed)
            np.random.seed(ev.seed)

        for pop in ev.populations:
            pop.initialize(seed=ev.seed)
            pop.evaluate(X, y[pop.name])

        pop_histories: dict[str, list[dict[str, Any]]] = {
            pop.name: [] for pop in ev.populations
        }
        interventions_left = self.max_interventions
        cooldown_remaining = 0

        for cb in ev.callbacks:
            cb.on_train_begin()

        try:
            for gen in range(ev.n_gen):
                for pop in ev.populations:
                    pop.apply_schedule(gen)
                    ev._evolve_population(pop, X, y[pop.name])

                if ev.cross_population and len(ev.populations) > 1:
                    ev._interpop_step(X, y)

                if (ev.migration and len(ev.populations) > 1
                        and gen > 0 and gen % ev.migration_freq == 0):
                    from symgene.operators.migration import migrate
                    migrate(
                        ev.populations,
                        topology=ev.migration_topology,
                        size=ev.migration_size,
                        selection=ev.migration_selection,
                        replace=ev.migration_replace,
                    )

                for pop in ev.populations:
                    pop.evaluate(X, y[pop.name])

                for pop in ev.populations:
                    best = pop.best
                    valid_fits = [
                        ind.fitness.values[0]
                        for ind in pop._population
                        if ind.fitness.valid and ind.fitness.values[0] < 1e6
                    ]
                    entry: dict[str, Any] = {
                        "gen": gen,
                        "train_mse": best.fitness.values[0] if best else 1e9,
                        "mean_train_mse": float(np.mean(valid_fits)) if valid_fits else 1e9,
                        "std_train_mse": float(np.std(valid_fits)) if valid_fits else 0.0,
                        "n_genes": len(best) if best else 0,
                    }
                    if X_val is not None and y_val is not None:
                        y_pred_val = ev._predict_individual(best, pop, X_val)
                        entry["val_r2"] = (
                            r2(y_val[pop.name], y_pred_val)
                            if y_pred_val is not None else None
                        )
                    pop_histories[pop.name].append(entry)

                if self.verbose >= 1 and gen % max(1, ev.n_gen // 20) == 0:
                    for pop in ev.populations:
                        fit = pop.best.fitness.values[0] if pop.best else "?"
                        print(f"Gen {gen:4d} | {pop.name} | fitness={fit:.6f}")

                # advisor hook
                if (interventions_left > 0
                        and cooldown_remaining == 0
                        and self._diagnostician is not None):
                    for pop in ev.populations:
                        val_fit: float | None = None
                        if X_val is not None and y_val is not None:
                            h = pop_histories[pop.name]
                            if h:
                                vr = h[-1].get("val_r2")
                                train = h[-1].get("train_mse", float("inf"))
                                if vr is not None:
                                    val_fit = float(1.0 - vr)

                        snapshot = self._monitor.update(
                            gen, pop, pop_histories[pop.name], val_fit
                        )
                        fired: BaseTrigger | None = None
                        for trigger in self.triggers:
                            if trigger.evaluate(snapshot):
                                fired = trigger
                                break

                        if fired is not None:
                            self._intervene(gen, pop, snapshot, fired)
                            interventions_left -= 1
                            cooldown_remaining = self.cooldown
                            break

                if cooldown_remaining > 0:
                    cooldown_remaining -= 1

                logs = ev._build_logs(gen, pop_histories)
                stop_signals = [cb.on_generation_end(gen, logs) for cb in ev.callbacks]
                if any(s is True for s in stop_signals):
                    if self.verbose >= 1:
                        print(f"  Early stopping at generation {gen}")
                    break

                if ev.checkpoint_dir and (gen + 1) % ev.checkpoint_every == 0:
                    ev._save_checkpoint(gen)

        finally:
            for cb in ev.callbacks:
                try:
                    cb.on_train_end()
                except Exception:
                    pass

        from symgene.results import SymGeneResult, PopulationResult
        result = SymGeneResult({
            pop.name: PopulationResult(pop, pop_histories[pop.name], ev)
            for pop in ev.populations
        })
        result.advisor_log_ = self._advisor_log
        return result

    def _intervene(
        self,
        gen: int,
        pop: Population,
        snapshot: Any,
        trigger: BaseTrigger,
    ) -> None:
        assert self._diagnostician is not None
        fitness_before = snapshot.best_fitness
        llm_calls = 0
        tokens_used = 0

        try:
            diagnosis = self._diagnostician.diagnose(snapshot)
            llm_calls += 1
            tokens_used += 800

            if self.verbose >= 1:
                self._print_block(gen, diagnosis)

            action_result = self._executor.execute(diagnosis, pop, gen)
            if diagnosis.recommended_action == "genetic_rescue":
                llm_calls += 1
                tokens_used += 1200

            self._advisor_log.append({
                "gen": gen,
                "trigger": trigger.name,
                "root_cause": diagnosis.root_cause,
                "action": diagnosis.recommended_action,
                "n_affected": action_result.get("n_affected", 0),
                "llm_calls": llm_calls,
                "tokens_used": tokens_used,
                "fitness_before": fitness_before,
                "status": "success",
            })

        except Exception as exc:
            if self.verbose >= 1:
                print(
                    f"\n[SymGene Advisor] WARNING: intervention failed "
                    f"at gen {gen}: {exc}\n"
                    "  Evolution continues normally.\n"
                )
            self._advisor_log.append({
                "gen": gen,
                "trigger": trigger.name,
                "status": "failed",
                "reason": str(exc),
            })

    def _print_block(self, gen: int, diagnosis: Diagnosis) -> None:
        print(f"\n{_BORDER}")
        print(f"  ◆ SymGene Advisor [Gen {gen}]")
        for line in textwrap.wrap(diagnosis.advisor_message, width=50):
            print(f"  {line}")
        print(f"{_BORDER}\n")
```

- [ ] **Step 4: Run tests to verify they pass**

```
python -m pytest tests/advisor/test_guided_evolver.py -v
```
Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```
git add symgene/advisor/guided_evolver.py tests/advisor/test_guided_evolver.py
git commit -m "feat(advisor): add AIGuidedEvolver with monitor, triggers, and LLM intervention"
```

---

## Task 6: Module Exports + Full Suite

**Files:**
- Create: `symgene/advisor/__init__.py`

- [ ] **Step 1: Create `symgene/advisor/__init__.py`**

```python
from symgene.advisor.guided_evolver import AIGuidedEvolver
from symgene.advisor.triggers import StagnationTrigger, DiversityTrigger, BloatTrigger
from symgene.advisor.monitor import EvolutionMonitor, MonitorSnapshot

__all__ = [
    "AIGuidedEvolver",
    "StagnationTrigger",
    "DiversityTrigger",
    "BloatTrigger",
    "EvolutionMonitor",
    "MonitorSnapshot",
]
```

- [ ] **Step 2: Run the full test suite**

```
python -m pytest tests/ -v --tb=short
```
Expected: all existing tests PASS + all new advisor tests PASS. Zero regressions.

- [ ] **Step 3: Verify separation — import advisor does not affect SymGeneEvolver**

```python
# Run manually in Python shell
from symgene import advisor          # import the new module
from symgene.evolver import SymGeneEvolver
import inspect
src = inspect.getsource(SymGeneEvolver.fit)
assert "advisor" not in src          # evolver has no advisor references
print("Separation confirmed.")
```

- [ ] **Step 4: Final commit**

```
git add symgene/advisor/__init__.py
git commit -m "feat(advisor): export public API — AIGuidedEvolver, triggers, monitor"
```

---

## Verification Checklist

- [ ] `SymGeneEvolver` produces identical results before and after importing `symgene.advisor`
- [ ] `AIGuidedEvolver(llm_client=None)` completes a full run with empty `advisor_log_`
- [ ] `StagnationTrigger(patience=1)` fires on the second generation in a stagnant run
- [ ] `ActionExecutor` with `client=None` returns `n_affected=0` for `genetic_rescue`
- [ ] `Diagnostician._parse_response` handles markdown fences without crashing
- [ ] Advisor block prints inline with generation logs when `verbose=1`
- [ ] All 276 existing tests still pass
