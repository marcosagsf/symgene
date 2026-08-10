"""Genetic Rescue — LLM-guided operator that rehabilitates the worst individuals."""
from __future__ import annotations

import random
import re
from typing import TYPE_CHECKING

import deap.gp as gp
import deap.tools as tools

if TYPE_CHECKING:
    from symgene.llm.client import LLMClient
    from symgene.llm.context import LLMContext
    from symgene.population import Population


_RESCUE_SYSTEM = """\
You are an expert in symbolic regression and genetic programming.
Your task: produce new mathematical gene expressions for underperforming individuals.

Rules:
- Use ONLY the primitives and variable names you are given
- Write in prefix notation: add(x1, mul(x2, sin(x3)))
- Every function call must have the correct number of arguments
- Do NOT add explanations, markdown, or code blocks — just the expression(s)"""

_RESCUE_USER_GENE = """\
Domain: {description}
Target: {target_name}
Available primitives: {primitives}
Variable names: {variable_names}

BEST individual (genes + fitted coefficients):
{best_genes}

WORST individual (genes + fitted coefficients):
{worst_genes}

Concepts describing successful structures:
{concepts}

Generate exactly ONE new gene expression to replace one gene of the worst individual.
It should borrow structure from the best individual but be distinct enough to escape
the current local optimum. Return ONLY the expression string in prefix notation."""

_RESCUE_USER_INDIVIDUAL = """\
Domain: {description}
Target: {target_name}
Available primitives: {primitives}
Variable names: {variable_names}

BEST individual (genes + fitted coefficients):
{best_genes}

WORST individual (genes + fitted coefficients):
{worst_genes}

Concepts describing successful structures:
{concepts}

Generate exactly {n_genes} new gene expressions to replace ALL genes of the worst individual.
Each gene must be distinct. Return EXACTLY {n_genes} expressions, ONE PER LINE, nothing else."""

_RETRY_SUFFIX = """\

RETRY — your previous response could not be parsed as a valid expression.
Check: prefix notation, correct arity, only allowed primitives and variable names.
Return ONLY the expression(s), one per line, no markdown."""


def _format_individual(ind) -> str:
    lines = []
    combiner = getattr(ind, "_combiner", None)
    coefs = getattr(combiner, "coef_", None) if combiner else None
    bias = getattr(combiner, "intercept_", None) if combiner else None

    for i, gene in enumerate(ind):
        coef_str = ""
        if coefs is not None and i < len(coefs):
            coef_str = f"  coef={float(coefs[i]):.6g}"
        lines.append(f"  gene[{i}]: {str(gene)}{coef_str}")

    if bias is not None:
        try:
            b = float(bias[0]) if hasattr(bias, "__len__") else float(bias)
            lines.append(f"  bias: {b:.6g}")
        except Exception:
            pass

    return "\n".join(lines) if lines else "  (empty)"


def _build_prompt(
    best_ind,
    worst_ind,
    ctx: "LLMContext",
    pop: "Population",
    n_genes: int = 1,
) -> str:
    primitives_list = ", ".join(name for _, _, name, *_ in pop.pset.primitives)
    variable_names = ", ".join(pop.pset.feature_names)
    best_str = _format_individual(best_ind)
    worst_str = _format_individual(worst_ind)
    concepts_str = (
        "\n".join(f"  - {c}" for c in ctx.sample(5))
        if not ctx.is_empty()
        else "  (no concepts yet)"
    )
    template = _RESCUE_USER_GENE if n_genes == 1 else _RESCUE_USER_INDIVIDUAL
    return template.format(
        description=ctx.description,
        target_name=ctx.target_name,
        primitives=primitives_list,
        variable_names=variable_names,
        best_genes=best_str,
        worst_genes=worst_str,
        concepts=concepts_str,
        n_genes=n_genes,
    )


def _clean_line(text: str) -> str:
    """Strip markdown, code fences, and leading list numbering."""
    text = re.sub(r"```[a-z]*\n?", "", text)
    text = text.strip().strip("`").strip()
    text = re.sub(r"^\d+[\.\)]\s*", "", text)
    return text.strip()


def _try_parse_gene(text: str, deap_pset, tree_max: int = 9999):
    """Parse one line of prefix-notation into a SGGene, or return None."""
    import deap.creator as creator

    text = _clean_line(text)
    if not text:
        return None
    try:
        tree = gp.PrimitiveTree.from_string(text, deap_pset)
        if len(tree) > tree_max:
            return None
        return creator.SGGene(tree)
    except Exception:
        return None


def _parse_response_single(response: str, deap_pset, tree_max: int):
    """Parse LLM response for gene-level rescue (one expression)."""
    for line in response.strip().splitlines():
        gene = _try_parse_gene(line, deap_pset, tree_max)
        if gene is not None:
            return gene
    return None


def _parse_response_multi(response: str, n_genes: int, deap_pset, tree_max: int):
    """Parse LLM response for individual-level rescue (N expressions)."""
    lines = [ln for ln in response.strip().splitlines() if ln.strip()]
    genes = [_try_parse_gene(ln, deap_pset, tree_max) for ln in lines]
    genes = [g for g in genes if g is not None]
    if not genes:
        return None
    while len(genes) < n_genes:
        genes.append(genes[-1])
    return genes[:n_genes]


def _call_llm_with_retry(
    client: "LLMClient",
    prompt_base: str,
    deap_pset,
    tree_max: int,
    max_retries: int,
    n_genes: int,
):
    """Call LLM with retry on parse failure. API exceptions propagate to caller."""
    prompt = prompt_base
    for attempt in range(max_retries):
        if attempt > 0:
            prompt = prompt_base + _RETRY_SUFFIX

        response = client.complete(prompt=prompt, system=_RESCUE_SYSTEM)

        if n_genes == 1:
            result = _parse_response_single(response, deap_pset, tree_max)
        else:
            result = _parse_response_multi(response, n_genes, deap_pset, tree_max)

        if result is not None:
            return result

    print(
        f"[SymGene LLM] Rescue: no valid expression after {max_retries} attempts — skipping."
    )
    return None


def rescue_worst(
    pop: "Population",
    client: "LLMClient",
    ctx: "LLMContext",
    rescue_fraction: float = 0.1,
    level: str = "gene",
    max_retries: int = 3,
) -> int:
    """Apply Genetic Rescue to the worst individuals in a population.

    Selects the worst ``rescue_fraction`` of the population by fitness and
    asks the LLM to produce new gene expression(s) for each, guided by the
    best individual's structure and the LLMContext concept library.

    Parameters
    ----------
    pop : Population
        Initialized population (``_population`` and ``_hof`` populated).
    client : LLMClient
        Configured LLM client. API exceptions propagate to the caller.
    ctx : LLMContext
        Concept library with domain description and target name.
    rescue_fraction : float
        Fraction of population to rescue (worst by fitness).
    level : str
        ``"gene"`` — replace one random gene per rescued individual.
        ``"individual"`` — replace all genes of each rescued individual.
    max_retries : int
        Retry attempts on format/parse errors before skipping one individual.

    Returns
    -------
    int
        Number of individuals whose genes were actually replaced.

    Raises
    ------
    Exception
        Any exception raised by ``client.complete()`` (API error) propagates
        to the caller so it can disable LLM for the remainder of the run.
    """
    population = pop._population
    deap_pset = pop._deap_pset
    best = pop.best

    if best is None or deap_pset is None or not population:
        return 0

    valid = [
        ind for ind in population
        if ind.fitness.valid and ind.fitness.values[0] < 1e8
    ]
    if len(valid) < 2:
        return 0

    n_rescue = max(1, int(len(population) * rescue_fraction))
    worst_inds = tools.selWorst(valid, min(n_rescue, len(valid) - 1))

    rescued = 0
    for worst in worst_inds:
        n_genes = len(worst)
        prompt_n = 1 if level == "gene" else n_genes
        prompt = _build_prompt(best, worst, ctx, pop, n_genes=prompt_n)

        result = _call_llm_with_retry(
            client, prompt, deap_pset, pop.tree_max, max_retries, n_genes=prompt_n
        )
        if result is None:
            continue

        if level == "gene":
            idx = random.randint(0, len(worst) - 1)
            worst[idx] = result
        else:
            for i, g in enumerate(result):
                if i < len(worst):
                    worst[i] = g

        if worst.fitness.valid:
            del worst.fitness.values
        rescued += 1

    return rescued
