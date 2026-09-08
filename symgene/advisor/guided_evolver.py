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

                # advisor hook — runs only when LLM client is present
                if (self._diagnostician is not None
                        and interventions_left > 0
                        and cooldown_remaining == 0):
                    for pop in ev.populations:
                        val_fit: float | None = None
                        if X_val is not None and y_val is not None:
                            h = pop_histories[pop.name]
                            if h:
                                vr = h[-1].get("val_r2")
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
        result.advisor_log_ = self._advisor_log  # type: ignore[attr-defined]
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
