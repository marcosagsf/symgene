from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from symgene.llm.client import LLMClient

# ---------------------------------------------------------------------------
# ConceptAbstraction — derive concepts from good vs bad expressions
# ---------------------------------------------------------------------------

_ABSTRACT_SYSTEM = """\
You are a helpful assistant that hypothesizes about the underlying mathematical
assumptions that generated a list of good and bad symbolic expressions.
Your goal is to identify what patterns distinguish the good expressions from the
bad ones, with special attention to mathematical structure and physical meaning.
Capital C represents an arbitrary constant in expressions."""

_ABSTRACT_USER = """\
Domain: {description}
Target variable: {target_name}
Input variables: {feature_names}

Good expressions (low prediction error):
{good_block}

Bad expressions (high prediction error):
{bad_block}

Propose {n} concise hypotheses about what mathematical patterns and structures
characterize the good expressions and exclude the bad ones.
Focus on mathematical character (e.g. "periodic structure", "ratio of variables",
"exponential decay"), not on complexity or simplicity.

End your response with a JSON list of hypothesis strings, e.g.:
["hypothesis 1", "hypothesis 2"]"""

# ---------------------------------------------------------------------------
# ConceptEvolution — refine and merge existing concepts
# ---------------------------------------------------------------------------

_EVOLVE_SYSTEM = """\
You are a helpful assistant that refines, merges, and generalizes hypotheses
about mathematical patterns in symbolic expressions.
Generate new insights that are more precise, surprising, or broadly applicable
than the originals. Do not simply restate existing hypotheses."""

_EVOLVE_USER = """\
Domain: {description}

Existing hypotheses about mathematical patterns:
{concepts_block}

Propose {n} new hypotheses that are refinements, generalizations, or creative
combinations of the existing ones. Aim for hypotheses that would guide a
symbolic regression algorithm toward better expressions.

End your response with a JSON list of hypothesis strings, e.g.:
["refined hypothesis 1", "refined hypothesis 2"]"""


def _extract_json_list(text: str) -> list[str]:
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON list found in LLM response:\n{text}")
    raw = json.loads(match.group())
    return [str(item) for item in raw if item]


def _format_expr_block(expressions: list[str]) -> str:
    return "\n".join(f"  {i + 1}. {expr}" for i, expr in enumerate(expressions))


def abstract_concepts(
    good_expressions: list[str],
    bad_expressions: list[str],
    client: "LLMClient",
    description: str = "not provided",
    target_name: str = "y",
    feature_names: list[str] | None = None,
    n_concepts: int = 3,
) -> list[str]:
    """Ask an LLM to hypothesize concepts from good vs bad expressions.

    Parameters
    ----------
    good_expressions : list of str
        String representations of high-fitness expressions.
    bad_expressions : list of str
        String representations of low-fitness expressions.
    client : LLMClient
        Configured LLM client.
    description : str
        Natural-language domain description.
    target_name : str
        Name of the target variable.
    feature_names : list of str or None
        Domain names of input variables.
    n_concepts : int
        Number of concepts to request.

    Returns
    -------
    list of str
        Generated concept strings. Empty list if LLM response cannot be parsed.
    """
    feat_str = ", ".join(feature_names) if feature_names else "not provided"
    prompt = _ABSTRACT_USER.format(
        description=description,
        target_name=target_name,
        feature_names=feat_str,
        good_block=_format_expr_block(good_expressions) or "  (none available)",
        bad_block=_format_expr_block(bad_expressions) or "  (none available)",
        n=n_concepts,
    )
    response = client.complete(prompt=prompt, system=_ABSTRACT_SYSTEM)
    try:
        return _extract_json_list(response)
    except ValueError:
        return []


def evolve_concepts(
    concepts: list[str],
    client: "LLMClient",
    description: str = "not provided",
    n_concepts: int = 3,
) -> list[str]:
    """Ask an LLM to refine and evolve an existing concept library.

    Parameters
    ----------
    concepts : list of str
        Current concept library to evolve.
    client : LLMClient
        Configured LLM client.
    description : str
        Natural-language domain description.
    n_concepts : int
        Number of evolved concepts to request.

    Returns
    -------
    list of str
        New concept strings. Empty list if LLM response cannot be parsed.
    """
    if not concepts:
        return []
    concepts_block = "\n".join(f"  {i + 1}. {c}" for i, c in enumerate(concepts))
    prompt = _EVOLVE_USER.format(
        description=description,
        concepts_block=concepts_block,
        n=n_concepts,
    )
    response = client.complete(prompt=prompt, system=_EVOLVE_SYSTEM)
    try:
        return _extract_json_list(response)
    except ValueError:
        return []
