from __future__ import annotations

import json
import re
import warnings
from typing import TYPE_CHECKING

from symgene.primitives.catalog import ALL, STANDARD

if TYPE_CHECKING:
    from symgene.llm.client import LLMClient


class InsufficientContextError(ValueError):
    """Raised when the LLM determines it needs more information to suggest primitives.

    The ``llm_message`` attribute contains the LLM's explanation of what
    additional context would be needed to make well-motivated suggestions.

    Examples
    --------
    >>> try:
    ...     pset = PrimitiveSet.from_description("model something", client, n_inputs=3)
    ... except InsufficientContextError as e:
    ...     print(e.llm_message)
    """

    def __init__(self, llm_message: str) -> None:
        self.llm_message = llm_message
        super().__init__(
            f"The LLM needs more information to suggest primitives.\n\n"
            f"{llm_message}\n\n"
            f"Provide a more detailed domain description and try again."
        )


_SYSTEM = """\
You are an expert in symbolic regression and mathematical modeling.
Your task is to select the most physically relevant mathematical primitives
for building symbolic expressions that model a described phenomenon.

You have exactly two valid response formats:

FORMAT A — when you have enough information:
Respond with ONLY a JSON list of primitive names, nothing else.
Example: ["add", "mul", "sin", "exp", "log"]

FORMAT B — when the description is too vague to make informed choices:
Respond with ONLY a JSON object with two keys, nothing else.
Example: {"needs_more_info": true, "message": "Describe what additional context is required."}

Never mix formats. Never add commentary outside the JSON."""

_USER = """\
Domain description:
{description}

Input variables: {feature_names}

Available primitives (name — description):
{primitives_table}

Select between {n_min} and {n_max} primitives most likely to appear in \
closed-form expressions that govern this phenomenon.
Consider the physical nature of the problem: prefer primitives whose \
mathematical character matches expected governing equations.

If the domain description is specific enough to motivate your choices,
respond with FORMAT A (JSON list).

If the description is too vague or generic (e.g. "model something", \
"predict a value") and you cannot make well-motivated selections,
respond with FORMAT B (JSON object with needs_more_info and message)."""

_PRIMITIVE_DESCRIPTIONS: dict[str, str] = {
    "add": "addition (a + b)",
    "sub": "subtraction (a - b)",
    "mul": "multiplication (a * b)",
    "div": "protected division (a / b)",
    "abs": "absolute value",
    "square": "square (x²)",
    "cube": "cube (x³)",
    "sqrt": "square root",
    "cbrt": "cube root",
    "inv": "inverse (1/x)",
    "pow": "power (a^b)",
    "sin": "sine",
    "cos": "cosine",
    "tan": "tangent",
    "asin": "arcsine",
    "acos": "arccosine",
    "atan": "arctangent",
    "exp": "exponential (eˣ)",
    "log": "natural logarithm",
    "log2": "base-2 logarithm",
    "log10": "base-10 logarithm",
    "tanh": "hyperbolic tangent",
    "sinh": "hyperbolic sine",
    "cosh": "hyperbolic cosine",
    "atanh": "inverse hyperbolic tangent",
    "sigmoid": "logistic sigmoid",
    "relu": "rectified linear unit",
    "gaussian": "Gaussian bell curve",
    "softplus": "smooth ReLU approximation",
    "softsign": "smooth sign function",
    "swish": "SiLU activation",
    "elu": "exponential linear unit",
    "sinc": "sinc function (sin(x)/x)",
    "mean2": "mean of 2 arguments",
    "mean3": "mean of 3 arguments",
    "mean4": "mean of 4 arguments",
    "max2": "max of 2 arguments",
    "max3": "max of 3 arguments",
    "max4": "max of 4 arguments",
    "min2": "min of 2 arguments",
    "min3": "min of 3 arguments",
    "min4": "min of 4 arguments",
    "harmonic_mean2": "harmonic mean of 2 arguments",
    "geometric_mean2": "geometric mean of 2 arguments",
    "if_positive": "conditional: if a > 0 return b else c",
    "if_greater": "conditional: if a > b return c else d",
    "step": "Heaviside step function",
}


def _build_primitives_table(names: list[str]) -> str:
    return "\n".join(
        f"  {name:<22} {_PRIMITIVE_DESCRIPTIONS.get(name, '')}"
        for name in names
    )


def _parse_response(text: str) -> list[str]:
    """Parse LLM response.

    Returns a list of primitive names if FORMAT A was used.
    Raises InsufficientContextError if FORMAT B was used.
    Raises ValueError if neither format was detected.
    """
    # Try FORMAT B first (needs_more_info object)
    obj_match = re.search(r"\{.*?\}", text, re.DOTALL)
    if obj_match:
        try:
            obj = json.loads(obj_match.group())
            if isinstance(obj, dict) and obj.get("needs_more_info"):
                message = obj.get("message", "No additional detail provided.")
                raise InsufficientContextError(message)
        except (json.JSONDecodeError, KeyError):
            pass  # not a valid needs_more_info object — fall through to list

    # Try FORMAT A (JSON list) — walk backwards through all '[' positions.
    # raw_decode tolerates trailing text and handles brackets inside strings.
    decoder = json.JSONDecoder()
    search_idx = len(text)
    while True:
        search_idx = text.rfind("[", 0, search_idx)
        if search_idx == -1:
            break
        try:
            parsed, _ = decoder.raw_decode(text, search_idx)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

    raise ValueError(f"LLM response did not match either expected format:\n{text}")


def suggest_primitives(
    description: str,
    client: "LLMClient",
    feature_names: list[str] | None = None,
    available: list[str] | None = None,
    n_min: int = 8,
    n_max: int = 16,
) -> list[str]:
    """Ask an LLM to select relevant primitives for a given domain.

    Parameters
    ----------
    description : str
        Natural-language description of the modeled phenomenon.
        The more specific, the better — e.g. "axial neutron flux in a
        pressurized water reactor with burnup and enrichment gradients"
        rather than "predict a physical quantity".
    client : LLMClient
        Configured LLM client (Anthropic or OpenAI).
    feature_names : list of str or None
        Domain names of input variables. These are passed to the LLM and
        strongly influence the quality of suggestions — always provide them.
    available : list of str or None
        Subset of the catalog to expose to the LLM.
        Defaults to the full ALL catalog.
    n_min, n_max : int
        Requested range for the number of primitives to select.

    Returns
    -------
    list of str
        Primitive names validated against the catalog.
        Unknown names suggested by the LLM are dropped with a warning.

    Raises
    ------
    InsufficientContextError
        If the LLM determines that the description is too vague to make
        well-motivated primitive selections. The exception's ``llm_message``
        attribute contains the LLM's explanation of what additional context
        is needed.
    """
    pool = available if available is not None else ALL
    feat_str = ", ".join(feature_names) if feature_names else "not provided"

    prompt = _USER.format(
        description=description,
        feature_names=feat_str,
        primitives_table=_build_primitives_table(pool),
        n_min=n_min,
        n_max=n_max,
    )

    response = client.complete(prompt=prompt, system=_SYSTEM)

    # May raise InsufficientContextError — let it propagate to the caller
    raw: list = _parse_response(response)

    # Validate names against the catalog
    valid_pool = set(pool)
    validated: list[str] = []
    for name in raw:
        if not isinstance(name, str):
            continue
        if name in valid_pool:
            validated.append(name)
        else:
            warnings.warn(
                f"LLM suggested unknown primitive '{name}' — skipped.",
                UserWarning,
                stacklevel=2,
            )

    if not validated:
        warnings.warn(
            "LLM returned no valid primitives. Falling back to STANDARD preset.",
            UserWarning,
            stacklevel=2,
        )
        return list(STANDARD)

    return validated
