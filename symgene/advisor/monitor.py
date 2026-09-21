from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np
import deap.gp as gp
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
    val_r2: float | None = None
    val_r2_history_last_10: list[float] = field(default_factory=list)
    n_genes_best: int = 0
    # enriched context for advisor
    top_individuals: list[dict] = field(default_factory=list)
    primitive_usage: dict[str, int] = field(default_factory=dict)
    current_params: dict[str, Any] = field(default_factory=dict)
    intervention_history: list[dict] = field(default_factory=list)


def _extract_top_individuals(pop: Any, k: int = 3) -> list[dict]:
    """Return top-k HoF individuals with gene expressions and Ridge coefficients."""
    hof = pop._hof
    if hof is None or len(hof) == 0:
        return []
    result = []
    for ind in list(hof)[:k]:
        combiner = getattr(ind, '_combiner', None)
        coefs = getattr(combiner, 'coef_', None)
        bias = getattr(combiner, 'bias_', None)
        genes_info = []
        for i, gene in enumerate(ind):
            info: dict[str, Any] = {"expr": str(gene), "len": len(gene)}
            if coefs is not None and i < len(coefs):
                info["coef"] = round(float(coefs[i]), 4)
            genes_info.append(info)
        entry: dict[str, Any] = {
            "fitness": round(float(ind.fitness.values[0]), 6) if ind.fitness.valid else None,
            "n_genes": len(ind),
            "genes": genes_info,
        }
        if bias is not None:
            try:
                b = float(bias[0]) if hasattr(bias, '__len__') else float(bias)
                entry["bias"] = round(b, 4)
            except Exception:
                pass
        result.append(entry)
    return result


def _count_primitive_usage(pop: Any, k: int = 10) -> dict[str, int]:
    """Count primitive occurrences across the top-k HoF individuals."""
    hof = pop._hof
    if hof is None:
        return {}
    counts: dict[str, int] = {}
    for ind in list(hof)[:k]:
        for gene in ind:
            for node in gene:
                if isinstance(node, gp.Primitive):
                    counts[node.name] = counts.get(node.name, 0) + 1
    return counts


def _get_current_params(pop: Any) -> dict[str, Any]:
    """Extract current evolvable parameters from the population."""
    params: dict[str, Any] = {
        "mutpb": round(pop.mutpb, 4),
        "cxpb": round(pop.cxpb, 4),
        "cxpb_low": round(pop.cxpb_low, 4),
        "elite_ratio": round(pop.elite_ratio, 4),
        "n_genes": pop.n_genes,
        "n_genes_max": pop.n_genes_max,
        "tree_max": pop.tree_max,
    }
    sel = pop.selection
    if hasattr(sel, 'size'):
        params["tournament_size"] = sel.size
    return params


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
        val_r2: float | None = None,
        intervention_history: list[dict] | None = None,
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

        overfit = float(val_fitness) if val_fitness is not None and history else None

        best_str = ""
        n_genes_best = 0
        if pop.best is not None:
            try:
                best_str = " | ".join(str(g) for g in pop.best)
                n_genes_best = len(pop.best)
            except Exception:
                best_str = str(pop.best)

        val_r2_hist = [
            h["val_r2"] for h in history[-10:]
            if "val_r2" in h and h["val_r2"] is not None
        ]

        top_individuals = _extract_top_individuals(pop, k=3)
        primitive_usage = _count_primitive_usage(pop, k=10)
        current_params = _get_current_params(pop)

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
            val_r2=val_r2,
            val_r2_history_last_10=val_r2_hist,
            n_genes_best=n_genes_best,
            top_individuals=top_individuals,
            primitive_usage=primitive_usage,
            current_params=current_params,
            intervention_history=(intervention_history or [])[-6:],
        )
