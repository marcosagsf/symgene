from __future__ import annotations
import json
import random
from typing import Any
import deap.gp as gp

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
                        if level == "gene":
                            replace_idx = random.randint(
                                0, len(pop._population[idx]) - 1
                            )
                            pop._population[idx][replace_idx] = tree
                        else:
                            n_genes = len(pop._population[idx])
                            new_genes = [
                                gp.PrimitiveTree.from_string(expr_str, pop._deap_pset)
                                for _ in range(n_genes)
                            ]
                            for gi, gene in enumerate(new_genes):
                                pop._population[idx][gi] = gene
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
