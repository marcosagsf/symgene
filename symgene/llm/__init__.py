from __future__ import annotations

__all__ = [
    "LLMClient",
    "LLMContext",
    "InsufficientContextError",
    "suggest_primitives",
    "interpret_population",
    "abstract_concepts",
    "evolve_concepts",
]


def __getattr__(name: str) -> object:
    if name == "LLMClient":
        from symgene.llm.client import LLMClient
        return LLMClient
    if name == "LLMContext":
        from symgene.llm.context import LLMContext
        return LLMContext
    if name == "InsufficientContextError":
        from symgene.llm.primitive_suggest import InsufficientContextError
        return InsufficientContextError
    if name == "suggest_primitives":
        from symgene.llm.primitive_suggest import suggest_primitives
        return suggest_primitives
    if name == "interpret_population":
        from symgene.llm.interpret import interpret_population
        return interpret_population
    if name == "abstract_concepts":
        from symgene.llm.concept_abstraction import abstract_concepts
        return abstract_concepts
    if name == "evolve_concepts":
        from symgene.llm.concept_abstraction import evolve_concepts
        return evolve_concepts
    raise AttributeError(f"module 'symgene.llm' has no attribute {name!r}")
