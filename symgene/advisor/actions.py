from __future__ import annotations
import json
import math as _math
import random
from typing import Any
import deap.gp as gp

from symgene.advisor.diagnostician import Diagnosis

_RESCUE_SYSTEM = (
    "You are SymGene Advisor. Generate new symbolic expressions in DEAP prefix notation "
    "to inject into the genetic programming population.\n"
    "STRICT RULES:\n"
    "- Use ONLY the names listed under 'Available DEAP functions' and 'Available terminals'\n"
    "- For variables: use the exact names listed under 'Variables' (e.g. Re, Pr, x0, x1)\n"
    "- For numeric constants: use ephemeral terminal names listed under 'Ephemerals' "
    "(e.g. pow_exp_0, lin_const_0) — NEVER write float literals like 0.5 or 2.0\n"
    "- Respond ONLY with a JSON array of strings, no markdown\n"
    "Example (functions=[add,mul,log,pow], variables=[Re,Pr], ephemerals=[pow_exp_0,lin_const_0]):\n"
    "[\"mul(log(Re), Pr)\", \"add(mul(Re, pow_exp_0), lin_const_0)\", "
    "\"pow(Re, pow_exp_0)\", \"mul(Re, add(Pr, lin_const_0))\"]\n"
    "Note: ephemerals receive distinct random numeric values for each individual at evaluation time."
)


def _fix_ephemerals(tree: gp.PrimitiveTree) -> gp.PrimitiveTree:
    """DEAP's from_string stores ephemeral *classes* (not instances) in the tree.
    Call each class to produce a Terminal instance with a sampled value.
    Without this, searchSubtree accesses Terminal.arity as a @property descriptor
    on the class itself (returns a property object, not 0) breaking 'while total > 0'.
    """
    for i, node in enumerate(tree):
        if isinstance(node, type):
            tree[i] = node()
    return tree


def _deap_fn_names(pop: Any) -> list[str]:
    try:
        return [
            name for name, obj in pop._deap_pset.mapping.items()
            if isinstance(obj, gp.Primitive)
        ]
    except Exception:
        return [pname for _, _, pname, *_ in pop.pset.primitives]


def _deap_var_names(pop: Any) -> list[str]:
    try:
        return [
            name for name, obj in pop._deap_pset.mapping.items()
            if isinstance(obj, gp.Terminal)
        ]
    except Exception:
        return list(pop.pset.feature_names) if hasattr(pop.pset, "feature_names") else []


def _deap_ephemeral_names(pop: Any) -> list[str]:
    try:
        return [
            name for name, obj in pop._deap_pset.mapping.items()
            if isinstance(obj, type) and hasattr(obj, "func")
        ]
    except Exception:
        return []


def _build_top_context(pop: Any, k: int = 3) -> str:
    """Format top-k HoF individuals with gene expressions and coefficients for the rescue prompt."""
    hof = pop._hof
    if hof is None or len(hof) == 0:
        return "(no evaluated individuals yet)"
    lines = []
    for rank, ind in enumerate(list(hof)[:k], 1):
        combiner = getattr(ind, '_combiner', None)
        coefs = getattr(combiner, 'coef_', None)
        bias = getattr(combiner, 'bias_', None)
        fit = f"{ind.fitness.values[0]:.6f}" if ind.fitness.valid else "?"
        lines.append(f"Rank {rank} | fitness={fit} | n_genes={len(ind)}")
        for i, gene in enumerate(ind):
            coef_str = ""
            if coefs is not None and i < len(coefs):
                coef_str = f"  [coef={float(coefs[i]):+.4f}]"
            lines.append(f"  gene[{i}]: {str(gene)}{coef_str}")
        if bias is not None:
            try:
                b = float(bias[0]) if hasattr(bias, '__len__') else float(bias)
                lines.append(f"  bias: {b:.4f}")
            except Exception:
                pass
    return "\n".join(lines)


class ActionExecutor:
    def __init__(self, client: Any | None = None) -> None:
        self.client = client

    def execute(self, diagnosis: Diagnosis, pop: Any, gen: int) -> dict[str, Any]:
        action = diagnosis.recommended_action
        if action == "genetic_rescue":
            return self._genetic_rescue(diagnosis, pop)
        if action in ("adjust_parameters", "adjust_mutation_rate"):
            return self._adjust_parameters(diagnosis, pop, action_name=action)
        if action == "force_migration":
            return {"action": "force_migration", "n_affected": 0}
        if action == "restart_worst":
            return self._restart_worst(diagnosis, pop)
        if action == "suggest_primitive":
            return self._suggest_primitive(diagnosis, pop)
        if action == "suggest_early_stop":
            return {"action": "suggest_early_stop", "n_affected": 0}
        return {"action": action, "n_affected": 0}

    def _genetic_rescue(self, diagnosis: Diagnosis, pop: Any) -> dict[str, Any]:
        if self.client is None:
            return {"action": "genetic_rescue", "n_affected": 0, "reason": "no llm client"}

        params = diagnosis.action_params
        fraction = float(params.get("rescue_fraction", 0.15))
        level = params.get("level", "individual")
        n_rescue = max(1, int(fraction * pop.pop_size))

        fn_names = _deap_fn_names(pop)
        var_names = _deap_var_names(pop)
        ephemeral_names = _deap_ephemeral_names(pop)
        top_context = _build_top_context(pop, k=3)

        eph_example = ephemeral_names[:2] if ephemeral_names else []

        prompt = (
            f"=== SYMBOLIC REGRESSION RESCUE: generate {n_rescue} new expressions ===\n\n"
            f"Available functions : {fn_names}\n"
            f"Variables           : {var_names}\n"
            f"Ephemerals          : {ephemeral_names}\n"
            f"  (Ephemerals are random numeric constants — always use them instead of float literals\n"
            f"   e.g. write pow_exp_0 not 0.8; lin_const_0 not 1000.0)\n\n"
            f"=== Top individuals (study structure + coefficients, then be creative) ===\n"
            f"{top_context}\n\n"
            f"=== Generation strategy ===\n"
            f"Genes with high absolute coefficient carry the most predictive power.\n"
            f"Generate {n_rescue} expressions that:\n"
            f"  1. Borrow structural motifs from the HIGH-COEF genes above\n"
            f"  2. Introduce enough variation to escape the current attractor\n"
            f"  3. Are syntactically valid DEAP prefix notation\n"
            f"  4. Use ephemerals like {eph_example} for any numeric constant\n"
            f"  5. Are structurally diverse from each other\n\n"
            f"Respond with a JSON array of exactly {n_rescue} valid expression strings."
        )

        worst_indices = sorted(
            range(len(pop._population)),
            key=lambda i: (
                pop._population[i].fitness.values[0]
                if pop._population[i].fitness.valid else float("inf")
            ),
            reverse=True,
        )[:n_rescue]

        rescued_set: set[int] = set()
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

                for idx, expr_str in zip(worst_indices, exprs):
                    if idx in rescued_set:
                        continue
                    try:
                        tree = _fix_ephemerals(
                            gp.PrimitiveTree.from_string(expr_str, pop._deap_pset)
                        )
                        if level == "gene":
                            replace_idx = random.randint(
                                0, len(pop._population[idx]) - 1
                            )
                            pop._population[idx][replace_idx] = tree
                        else:
                            n_genes_ind = len(pop._population[idx])
                            for gi in range(n_genes_ind):
                                pop._population[idx][gi] = _fix_ephemerals(
                                    gp.PrimitiveTree.from_string(
                                        expr_str, pop._deap_pset
                                    )
                                )
                        del pop._population[idx].fitness.values
                        rescued_set.add(idx)
                    except Exception:
                        continue
                break
            except Exception:
                continue

        llm_injected = len(rescued_set)

        unrescued = [idx for idx in worst_indices if idx not in rescued_set]
        if unrescued and pop._toolbox is not None:
            for idx in unrescued:
                try:
                    n_genes_ind = len(pop._population[idx])
                    for gi in range(n_genes_ind):
                        pop._population[idx][gi] = pop._toolbox.gene()
                    del pop._population[idx].fitness.values
                    rescued_set.add(idx)
                except Exception:
                    continue

        rescued = len(rescued_set)
        return {
            "action": "genetic_rescue",
            "n_affected": rescued,
            "llm_injected": llm_injected,
        }

    def _adjust_parameters(self, diagnosis: Diagnosis, pop: Any, action_name: str = "adjust_parameters") -> dict[str, Any]:
        params = diagnosis.action_params
        # legacy alias: adjust_mutation_rate uses new_mutpb
        if "new_mutpb" in params and "mutpb" not in params:
            params = dict(params, mutpb=params["new_mutpb"])

        changes: dict[str, Any] = {}

        if "mutpb" in params:
            val = max(0.05, min(0.5, float(params["mutpb"])))
            old = pop.mutpb
            pop.mutpb = val
            changes["mutpb"] = {"from": round(old, 4), "to": round(val, 4)}

        if "cxpb" in params:
            val = max(0.5, min(0.98, float(params["cxpb"])))
            old = pop.cxpb
            pop.cxpb = val
            changes["cxpb"] = {"from": round(old, 4), "to": round(val, 4)}

        if "cxpb_low" in params:
            val = max(0.1, min(0.9, float(params["cxpb_low"])))
            old = pop.cxpb_low
            pop.cxpb_low = val
            changes["cxpb_low"] = {"from": round(old, 4), "to": round(val, 4)}

        if "tournament_size" in params:
            size = max(2, min(10, int(params["tournament_size"])))
            if hasattr(pop.selection, 'size'):
                old = pop.selection.size
                pop.selection.size = size
                changes["tournament_size"] = {"from": old, "to": size}

        if "elite_ratio" in params:
            val = max(0.01, min(0.10, float(params["elite_ratio"])))
            old = pop.elite_ratio
            pop.elite_ratio = val
            changes["elite_ratio"] = {"from": round(old, 4), "to": round(val, 4)}

        return {"action": action_name, "n_affected": len(changes), "changes": changes}

    def _suggest_primitive(self, diagnosis: Diagnosis, pop: Any) -> dict[str, Any]:
        params = diagnosis.action_params
        name = str(params.get("name", "")).strip().replace(" ", "_")
        lambda_str = str(params.get("lambda_str", "")).strip()
        arity_val = max(1, min(2, int(params.get("arity", 1))))
        description = str(params.get("description", ""))
        remove_suggestion = params.get("remove_primitive")

        if not name or not lambda_str:
            return {"action": "suggest_primitive", "n_affected": 0,
                    "reason": "missing 'name' or 'lambda_str' in action_params"}

        # Safe evaluation context — only math and safe builtins
        _safe_ctx = {
            "math": _math,
            "abs": abs, "min": min, "max": max, "pow": pow,
            "log": _math.log, "sqrt": _math.sqrt, "exp": _math.exp,
            "sin": _math.sin, "cos": _math.cos, "fabs": _math.fabs,
        }
        try:
            fn_raw = eval(lambda_str, {"__builtins__": {}}, _safe_ctx)
            if not callable(fn_raw):
                raise ValueError("eval result is not callable")
            # Test on safe positive values to catch domain errors
            _test_vals = [2.0, 3.5]
            fn_raw(*_test_vals[:arity_val])
        except Exception as exc:
            return {"action": "suggest_primitive", "n_affected": 0,
                    "reason": f"lambda eval/test failed: {exc}"}

        # Avoid name collisions
        existing_names = {p[2] for p in pop.pset.primitives}
        if name in existing_names:
            name = f"{name}_new"

        # Wrap with squash + error guard for numerical safety
        squash = pop.pset.squash
        _fn = fn_raw
        if squash is not None:
            if arity_val == 1:
                def fn_final(x, _f=_fn, _s=squash):
                    try:
                        return _s(_f(x))
                    except Exception:
                        return 0.0
            else:
                def fn_final(x, y, _f=_fn, _s=squash):
                    try:
                        return _s(_f(x, y))
                    except Exception:
                        return 0.0
        else:
            if arity_val == 1:
                def fn_final(x, _f=_fn):
                    try:
                        return float(_f(x))
                    except Exception:
                        return 0.0
            else:
                def fn_final(x, y, _f=_fn):
                    try:
                        return float(_f(x, y))
                    except Exception:
                        return 0.0
        fn_final.__name__ = name

        # Register in symgene pset (for display and future serialization)
        pop.pset.primitives.append((fn_final, arity_val, name, None))

        # Register in DEAP pset — future tree generation & mutation will include it
        # (existing individuals are unaffected; they reference their own node objects)
        try:
            pop._deap_pset.addPrimitive(fn_final, arity_val, name=name)
        except Exception as exc:
            pop.pset.primitives.pop()  # roll back
            return {"action": "suggest_primitive", "n_affected": 0,
                    "reason": f"DEAP addPrimitive failed: {exc}"}

        return {
            "action": "suggest_primitive",
            "n_affected": 1,
            "name": name,
            "arity": arity_val,
            "description": description,
            "lambda_str": lambda_str,
            "remove_suggestion": remove_suggestion,
        }

    def _adjust_mutation_rate(self, diagnosis: Diagnosis, pop: Any) -> dict[str, Any]:
        # Legacy shim — delegates to adjust_parameters
        return self._adjust_parameters(diagnosis, pop)

    def _restart_worst(self, diagnosis: Diagnosis, pop: Any) -> dict[str, Any]:
        fraction = float(diagnosis.action_params.get("fraction", 0.2))
        n_restart = max(1, int(fraction * pop.pop_size))
        worst_indices = sorted(
            range(len(pop._population)),
            key=lambda i: (
                pop._population[i].fitness.values[0]
                if pop._population[i].fitness.valid else float("inf")
            ),
            reverse=True,
        )[:n_restart]
        restarted = 0
        if pop._toolbox is not None:
            for idx in worst_indices:
                try:
                    n_genes_ind = len(pop._population[idx])
                    for gi in range(n_genes_ind):
                        pop._population[idx][gi] = pop._toolbox.gene()
                    del pop._population[idx].fitness.values
                    restarted += 1
                except Exception:
                    continue
        return {"action": "restart_worst", "n_affected": restarted}
