from __future__ import annotations
import random
import textwrap
import time
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

# Extra cooldown penalty applied after an LLM rate-limit (429) error,
# expressed in generations. Helps avoid hammering rate-limited APIs.
_RATE_LIMIT_COOLDOWN_PENALTY = 30


class AIGuidedEvolver:
    def __init__(
        self,
        populations: list[Population],
        n_gen: int = 200,
        llm_client: Any = None,
        triggers: list[BaseTrigger] | None = None,
        max_interventions: int = 8,
        cooldown: int = 15,
        val_r2_threshold: float = 0.999,
        wait_on_rate_limit: bool = False,
        rate_limit_wait: int = 65,
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
        # Skip advisor when validation R² already exceeds this threshold.
        # Set to 0.0 to disable (always allow interventions).
        self.val_r2_threshold = val_r2_threshold
        # If True: pause evolution and wait rate_limit_wait seconds before
        # retrying (counts as 1 intervention). If False: skip and cooldown.
        self.wait_on_rate_limit = wait_on_rate_limit
        self.rate_limit_wait = rate_limit_wait
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

        if self.verbose >= 1:
            if self._diagnostician is None:
                print(
                    "[AIGuidedEvolver] Dry-run mode: llm_client=None "
                    "— triggers are evaluated but no LLM intervention occurs."
                )
            else:
                _prov = getattr(self.llm_client, 'provider', '?')
                print(
                    f"[AIGuidedEvolver] Advisor active | provider={_prov} | "
                    f"max_interventions={self.max_interventions} | "
                    f"cooldown={self.cooldown} | "
                    f"val_r2_threshold={self.val_r2_threshold}"
                )

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

                # advisor hook — triggers always evaluated; LLM called only when client is set
                if interventions_left > 0 and cooldown_remaining == 0:
                    for pop in ev.populations:
                        h = pop_histories[pop.name]
                        val_r2_now: float | None = None
                        val_fit: float | None = None
                        if X_val is not None and y_val is not None and h:
                            vr = h[-1].get("val_r2")
                            if vr is not None:
                                val_r2_now = float(vr)
                                val_fit = float(1.0 - vr)

                        # skip when model has already converged
                        if (self.val_r2_threshold > 0.0
                                and val_r2_now is not None
                                and val_r2_now >= self.val_r2_threshold):
                            continue

                        snapshot = self._monitor.update(
                            gen, pop, h, val_fit, val_r2=val_r2_now,
                            intervention_history=self._advisor_log,
                        )
                        fired: BaseTrigger | None = None
                        for trigger in self.triggers:
                            if trigger.evaluate(snapshot):
                                fired = trigger
                                break

                        if fired is not None:
                            if self._diagnostician is not None:
                                penalty = self._intervene(gen, pop, snapshot, fired)
                                interventions_left -= 1
                                cooldown_remaining = self.cooldown + penalty
                                # Re-evaluate after intervention: genetic_rescue sets
                                # fitness.valid = False on replaced individuals.
                                pop.evaluate(X, y[pop.name])
                            else:
                                # dry-run: trigger detected but no LLM available
                                vr_str = f"{val_r2_now:.4f}" if val_r2_now is not None else "N/A"
                                if self.verbose >= 1:
                                    print(
                                        f"\n[AIGuidedEvolver] Trigger: {fired.name} "
                                        f"at gen {gen} | val_r2={vr_str} "
                                        f"— dry-run (no llm_client, no action taken)."
                                    )
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
        result.advisor_log_ = self._advisor_log  # type: ignore[attr-defined]
        return result

    def _intervene(
        self,
        gen: int,
        pop: Population,
        snapshot: Any,
        trigger: BaseTrigger,
    ) -> int:
        """Execute one LLM-guided intervention. Returns extra cooldown penalty (0 normally, >0 on rate limit)."""
        assert self._diagnostician is not None
        fitness_before = snapshot.best_fitness
        llm_calls = 0
        tokens_used = 0

        # Inner attempt — separated so rate-limit retry can call it again.
        def _attempt() -> tuple[bool, int]:
            nonlocal llm_calls, tokens_used
            try:
                diagnosis = self._diagnostician.diagnose(snapshot)
                llm_calls += 1
                tokens_used += 800

                if self.verbose >= 1:
                    self._print_block(gen, diagnosis, trigger)

                action_result = self._executor.execute(diagnosis, pop, gen)
                if diagnosis.recommended_action == "genetic_rescue":
                    llm_calls += 1
                    tokens_used += 1200

                if self.verbose >= 1:
                    self._print_action_result(diagnosis, action_result, snapshot)

                self._advisor_log.append({
                    "gen": gen,
                    "trigger": trigger.name,
                    "root_cause": diagnosis.root_cause,
                    "action": diagnosis.recommended_action,
                    "n_affected": action_result.get("n_affected", 0),
                    "changes": action_result.get("changes", {}),
                    "llm_calls": llm_calls,
                    "tokens_used": tokens_used,
                    "fitness_before": fitness_before,
                    "reasoning": diagnosis.reasoning[:300] if diagnosis.reasoning else "",
                    "status": "success",
                })
                return True, 0

            except Exception as exc:
                exc_str = str(exc)
                exc_type = type(exc).__name__
                is_rate_limit = "429" in exc_str or "RESOURCE_EXHAUSTED" in exc_str
                penalty = _RATE_LIMIT_COOLDOWN_PENALTY if is_rate_limit else 0

                is_quota_exhausted = "free_tier_requests" in exc_str or (
                    is_rate_limit and "free_tier" in exc_str
                )

                if self.verbose >= 1:
                    if is_quota_exhausted:
                        print(
                            f"\n[SymGene Advisor] QUOTA EXHAUSTED (gen {gen}): "
                            f"generativelanguage.googleapis.com/generate_content_free_tier_requests.\n"
                            f"  This is not a per-minute limit — it is the total free-tier quota.\n"
                            f"  Options: (1) wait for the daily reset (midnight UTC), "
                            f"(2) enable billing on Google Cloud, "
                            f"(3) use llm_client=None for a dry-run.\n"
                            f"  Advisor disabled for {penalty} generations.\n"
                        )
                    elif is_rate_limit:
                        print(
                            f"\n[SymGene Advisor] WARNING: rate limit (RPM) at gen {gen} "
                            f"— waiting a cooldown of {penalty} gens.\n"
                            f"  [{exc_type}] {exc_str[:200]}\n"
                        )
                    else:
                        print(
                            f"\n[SymGene Advisor] WARNING: intervention failed "
                            f"at gen {gen}: [{exc_type}] {exc_str[:300]}\n"
                            "  Evolution continues normally.\n"
                        )

                self._advisor_log.append({
                    "gen": gen,
                    "trigger": trigger.name,
                    "status": "rate_limited" if is_rate_limit else "failed",
                    "reason": exc_str,
                    "extra_cooldown": penalty,
                })
                return False, penalty

        success, penalty = _attempt()
        if success:
            return 0

        # If rate-limited and wait_on_rate_limit=True: pause and retry once.
        is_rate_limit = penalty > 0
        if is_rate_limit and self.wait_on_rate_limit:
            wait_sec = self.rate_limit_wait
            if self.verbose >= 1:
                print(
                    f"  [SymGene Advisor] Waiting {wait_sec}s before retrying "
                    f"(wait_on_rate_limit=True)..."
                )
            time.sleep(wait_sec)
            success2, penalty2 = _attempt()
            if success2:
                if self.verbose >= 1:
                    print("  [SymGene Advisor] Retry succeeded after waiting.\n")
                return 0
            if self.verbose >= 1:
                print("  [SymGene Advisor] Retry failed. Cooldown applied.\n")
            return penalty2

        if is_rate_limit and self.verbose >= 1:
            print(
                f"  Adding {penalty}-gen cooldown. Evolution continues normally.\n"
            )
        return penalty

    def _print_block(self, gen: int, diagnosis: Diagnosis, trigger: BaseTrigger) -> None:
        print(f"\n{_BORDER}")
        print(f"  ◆ SymGene Advisor [Gen {gen}] | Trigger: {trigger.name}")
        print(f"  Cause: {diagnosis.root_cause} | Action: {diagnosis.recommended_action}")
        for line in textwrap.wrap(diagnosis.advisor_message, width=50):
            print(f"  {line}")
        print(f"{_BORDER}")

    def _print_action_result(
        self, diagnosis: Diagnosis, action_result: dict, snapshot: Any
    ) -> None:
        action = diagnosis.recommended_action
        n = action_result.get("n_affected", 0)
        llm_n = action_result.get("llm_injected", n)
        val_r2_str = f"{snapshot.val_r2:.4f}" if snapshot.val_r2 is not None else "N/A"

        if action == "genetic_rescue":
            if n == 0:
                body = (
                    f"Genetic rescue FAILED: no individual replaced. "
                    f"This may mean the LLM produced expressions with invalid tokens. "
                    f"Evolution continues normally."
                )
            else:
                fallback_n = n - llm_n
                parts = []
                if llm_n > 0:
                    parts.append(f"{llm_n} expression(s) injected by the LLM")
                if fallback_n > 0:
                    parts.append(f"{fallback_n} regenerated randomly by the toolbox")
                body = (
                    f"Genetic rescue complete: {n} individual(s) replaced "
                    f"({'; '.join(parts)}). The worst {n} individuals were replaced "
                    f"with new structures. Validation R²: {val_r2_str}."
                )
        elif action in ("adjust_parameters", "adjust_mutation_rate"):
            changes = action_result.get("changes", {})
            if changes:
                change_strs = [
                    f"{k}: {v['from']}→{v['to']}" for k, v in changes.items()
                ]
                body = (
                    f"Evolutionary parameters adjusted: {', '.join(change_strs)}. "
                    f"Validation R²: {val_r2_str}."
                )
            else:
                body = f"No parameter changed (values already at their limits). R²={val_r2_str}."
        elif action == "suggest_primitive":
            if n > 0:
                pname = action_result.get("name", "?")
                desc = action_result.get("description", "")
                lstr = action_result.get("lambda_str", "")[:60]
                body = (
                    f"New primitive '{pname}' (arity={action_result.get('arity', '?')}) "
                    f"added to the pset: {lstr}. "
                    f"{desc[:80]} R²={val_r2_str}."
                )
            else:
                reason = action_result.get("reason", "unknown error")
                body = f"Primitive suggestion FAILED: {reason}. R²={val_r2_str}."
        elif action == "restart_worst":
            body = (
                f"Restarted the {n} worst individuals with random expressions. "
                f"Validation R²: {val_r2_str}."
            )
        elif action == "suggest_early_stop":
            body = f"Early stop suggested — model converged (R²={val_r2_str})."
        else:
            body = f"Action '{action}': {n} element(s) affected. R²={val_r2_str}."

        for line in textwrap.wrap(f"  >> {body}", width=54):
            print(line)
        print()
