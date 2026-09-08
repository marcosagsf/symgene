# Design: AIGuidedEvolver — LLM-Assisted Evolutionary Guidance

**Date:** 2026-09-08
**Status:** Approved for implementation
**Module:** `symgene/advisor/`

---

## Overview

`AIGuidedEvolver` is an optional wrapper around the existing `SymGeneEvolver` that adds
LLM-guided intervention during evolution. It monitors the population continuously (zero LLM
cost) and calls the LLM only when specific trigger conditions are met — diagnosing the
problem and executing a targeted action.

The existing `SymGeneEvolver` is not modified. Users opt in explicitly.

---

## Core Principles

1. **Separation**: `symgene/advisor/` is a standalone module. Zero changes to existing code.
2. **Optionality**: `SymGeneEvolver` continues to work identically without the advisor.
3. **Cost control**: LLM is called sparingly — estimated 3–10 calls per run, never every generation.
4. **Robustness**: LLM failure never aborts a run. Evolution always completes.
5. **Transparency**: Every LLM intervention is printed inline during training and logged in the result.

---

## Architecture

```
symgene/
  evolver.py            ← unchanged
  population.py         ← unchanged
  results.py            ← unchanged
  advisor/              ← NEW MODULE
    __init__.py
    monitor.py          ← EvolutionMonitor (zero LLM cost)
    triggers.py         ← StagnationTrigger, DiversityTrigger, BloatTrigger
    diagnostician.py    ← LLM diagnosis call (Stage 1)
    actions.py          ← Action definitions + executors (deterministic + LLM Stage 2)
    guided_evolver.py   ← AIGuidedEvolver wrapper
```

`AIGuidedEvolver` wraps `SymGeneEvolver` by composition, not inheritance. It drives the
generation loop internally, delegating per-generation evolution to the wrapped evolver's
existing methods (`_evolve_population`, `evaluate`, etc.).

---

## User API

```python
# Traditional usage — UNCHANGED
from symgene import SymGeneEvolver
evolver = SymGeneEvolver(populations=[pop], n_gen=200)
result = evolver.fit(X, y)

# AI-guided usage — NEW, explicit opt-in
from symgene.advisor import AIGuidedEvolver, StagnationTrigger, DiversityTrigger
from symgene.llm import LLMClient

client = LLMClient(provider="anthropic", model="claude-haiku-4-5-20251001")

guided = AIGuidedEvolver(
    populations=[pop],
    n_gen=200,
    llm_client=client,
    triggers=[
        StagnationTrigger(patience=20),
        DiversityTrigger(threshold=0.15),
        BloatTrigger(complexity_growth=0.10),
    ],
    max_interventions=8,   # max LLM calls per run
    cooldown=15,           # min generations between interventions
    verbose=1,             # same as SymGeneEvolver
)
result = guided.fit(X, y)  # returns identical SymGeneResult + advisor_log_
```

If `triggers` is omitted, sensible defaults are applied automatically.
If `llm_client=None`, the advisor monitors but never intervenes (dry-run mode).

---

## Module: monitor.py — EvolutionMonitor

Tracks population signals every generation. No LLM cost.

| Signal | Calculation |
|---|---|
| `stagnation_counter` | Generations without improvement in best fitness |
| `diversity_index` | Fraction of unique individuals (via existing `metrics/diversity.py`) |
| `complexity_trend` | Average node growth over last 10 generations with no fitness gain |
| `overfit_gap` | `val_mse - train_mse` delta (only when X_val provided) |
| `fitness_variance` | Variance of fitness values across current population |

The monitor builds a compact `MonitorSnapshot` dataclass per generation, passed to triggers
and later to the diagnostician.

---

## Module: triggers.py — Trigger Conditions

Each trigger evaluates a `MonitorSnapshot` and returns `True` when its condition is met.

```python
StagnationTrigger(patience: int = 20)
    # fires when stagnation_counter >= patience

DiversityTrigger(threshold: float = 0.15)
    # fires when diversity_index < threshold

BloatTrigger(complexity_growth: float = 0.10)
    # fires when complexity grew > 10% over last 10 gens with no fitness gain
```

**Dispatch rules:**
- Only one trigger fires per generation — the one with highest priority (declaration order).
- After an intervention, `cooldown` generations are blocked for all triggers.
- When `max_interventions` is reached, triggers stop firing. Evolution continues normally.

---

## Module: diagnostician.py — Stage 1 LLM Call

**Input** (compact, not full history):
```json
{
  "trigger": "StagnationTrigger",
  "stagnation_counter": 23,
  "diversity_index": 0.09,
  "complexity_trend": 0.08,
  "best_fitness": 0.0241,
  "best_expression_latex": "\\sin(x_1) \\cdot e^{-x_2^2}",
  "fitness_history_last_30": [0.0241, 0.0241, 0.0242, ...],
  "available_primitives": ["add", "mul", "sin", "log", "exp", ...]
}
```

**Output** (JSON schema, validated via structured outputs API — no regex):
```json
{
  "root_cause": "premature_convergence",
  "severity": "high",
  "confidence": 0.88,
  "recommended_action": "genetic_rescue",
  "action_params": {"rescue_fraction": 0.15, "level": "individual"},
  "advisor_message": "Stagnation detected over 23 generations. Genetic diversity collapsed..."
}
```

`advisor_message` is the human-readable text printed to screen and saved to the log.

---

## Module: actions.py — Stage 2 Execution

Based on `recommended_action`, the executor acts. Most actions are deterministic (no second
LLM call). Only `genetic_rescue` requires a second LLM call to generate new gene expressions.

| Action | Executor | Second LLM call? |
|---|---|---|
| `genetic_rescue` | Generate new individuals in DEAP prefix notation | ✅ yes |
| `adjust_mutation_rate` | Set `pop.mutpb` directly | ❌ no |
| `force_migration` | Call existing `migrate()` | ❌ no |
| `add_primitive` | Add primitive from catalog | ❌ no |
| `suggest_early_stop` | Set flag in log + print warning | ❌ no |
| `restart_worst` | Reinitialize worst X% of population | ❌ no |

For `genetic_rescue`, the second LLM call receives the diagnosis + current best individual
and returns new gene expressions in DEAP prefix notation (same format as existing Phase 3).

---

## Visual Output During Training

When an intervention occurs, a clearly delimited block is printed inline with generation logs:

```
Gen  045 | _target | fitness=0.024103
Gen  046 | _target | fitness=0.024098
Gen  047 | _target | fitness=0.024101

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ◆ SymGene Advisor [Gen 47]
  Stagnation detected over 23 generations. Analyzing
  population dynamics and expression landscape...

  Diagnosis: Premature convergence — diversity collapsed
  to 9%. Population is cycling through variations of
  the same structural pattern.

  Action: Injecting 8 new individuals with structurally
  diverse expressions. New candidates explore logarithmic
  and rational forms absent from current population.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Gen  048 | _target | fitness=0.019823
```

**Output rules:**
- Always inline with generation prints — not in a separate log file.
- Three fixed parts: what was detected, what was diagnosed, what was done.
- Affirmative language only — no questions to user.
- Suppressed when `verbose=0`.

---

## Result: advisor_log_

`SymGeneResult` gains one new optional attribute `advisor_log_` (None if AIGuidedEvolver
was not used). It is a list of intervention records:

```python
result.advisor_log_ = [
    {
        "gen": 47,
        "trigger": "StagnationTrigger",
        "root_cause": "premature_convergence",
        "action": "genetic_rescue",
        "n_rescued": 8,
        "llm_calls": 2,
        "tokens_used": 1840,
        "fitness_before": 0.024101,
        "fitness_after": 0.019823,   # next gen
    },
    ...
]
```

---

## Error Handling

LLM failure never aborts a run:

| Failure | Behavior |
|---|---|
| API timeout / error | Warning printed, intervention skipped, evolution continues |
| Invalid JSON response | Retry up to 2x, then skip |
| `max_interventions` reached | Monitor continues, LLM never called again |
| `llm_client=None` | Dry-run: monitor tracks but never intervenes |

All failures are recorded in `advisor_log_` with `"status": "failed"` and `"reason"`.

---

## Test Plan (POC)

Three tests to validate the POC:

1. **Separation test**: `SymGeneEvolver` produces identical results whether or not
   `symgene.advisor` is imported.

2. **Trigger fires**: a short run with `StagnationTrigger(patience=5)` correctly detects
   stagnation and calls the diagnostician.

3. **Graceful degradation**: `AIGuidedEvolver(llm_client=None)` completes a full run
   without error, with `advisor_log_` empty.

---

## Out of Scope (POC)

- Multi-population advisor coordination (each population advised independently for now)
- Persistent concept memory across runs (LLMContext not integrated in this phase)
- User-configurable LLM prompt templates
- Token budget pre-estimation before calling

These are natural extensions for after the POC validates the core loop.
