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
            overfit = float(val_fitness)

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
