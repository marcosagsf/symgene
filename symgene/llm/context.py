from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from symgene.llm.client import LLMClient
    from symgene.results import PopulationResult


def _individual_to_str(ind) -> str:
    """Convert a MGGP individual (list of genes) to a readable string."""
    return " + ".join(str(gene) for gene in ind)


def _hof_expressions(pop_result: "PopulationResult", n_good: int, n_bad: int):
    """Return (good_exprs, bad_exprs) string lists from a PopulationResult's HOF."""
    hof = list(pop_result._pop._hof)
    if not hof:
        best = pop_result.best_individual_
        good = [_individual_to_str(best)] if best is not None else []
        return good, []

    # HOF in DEAP is sorted best-first (lowest fitness value = better)
    good = [_individual_to_str(ind) for ind in hof[:n_good]]
    bad = [_individual_to_str(ind) for ind in hof[max(0, len(hof) - n_bad):]]
    return good, bad


@dataclass
class LLMContext:
    """Library of natural-language concepts that guide LLM-assisted evolution.

    Concepts are short strings describing mathematical or physical patterns
    observed in the best (and worst) expressions found during evolution.
    They are passed as context in mutation and crossover prompts so the LLM
    can reason about what structures work well for the problem.

    Parameters
    ----------
    concepts : list of str
        Current concept library. Grows via ``from_result()`` or ``evolve()``.
    variable_names : dict of str to str
        Maps internal variable identifiers (e.g. ``"x0"``) to domain names
        (e.g. ``"burnup"``). Domain names are essential: without them the LLM
        cannot activate physical knowledge.
    max_concepts : int
        Maximum library size. Oldest concepts are dropped when exceeded.
    description : str
        Domain description carried with the context for use in prompts.
    target_name : str
        Target variable name carried with the context.
    """

    concepts: list[str] = field(default_factory=list)
    variable_names: dict[str, str] = field(default_factory=dict)
    max_concepts: int = 20
    description: str = "not provided"
    target_name: str = "y"

    # ------------------------------------------------------------------
    # Basic library management
    # ------------------------------------------------------------------

    def add(self, concept: str) -> None:
        """Add a concept, dropping the oldest entry if over capacity."""
        self.concepts.append(concept)
        if len(self.concepts) > self.max_concepts:
            self.concepts.pop(0)

    def add_many(self, concepts: list[str]) -> None:
        """Add multiple concepts."""
        for c in concepts:
            self.add(c)

    def sample(self, n: int = 3) -> list[str]:
        """Return up to n concepts sampled from the most recent entries.

        Sampling is done without replacement from the last ``max_concepts``
        entries so that very old concepts are naturally phased out.
        """
        recent = self.concepts[-self.max_concepts:]
        return random.sample(recent, min(n, len(recent)))

    def is_empty(self) -> bool:
        return len(self.concepts) == 0

    def __len__(self) -> int:
        return len(self.concepts)

    def __repr__(self) -> str:
        return (
            f"LLMContext(n_concepts={len(self.concepts)}, "
            f"target={self.target_name!r}, "
            f"max_concepts={self.max_concepts})"
        )

    # ------------------------------------------------------------------
    # Building context from a fitted PopulationResult
    # ------------------------------------------------------------------

    @classmethod
    def from_result(
        cls,
        result: "PopulationResult",
        client: "LLMClient",
        description: str = "not provided",
        target_name: str | None = None,
        variable_names: dict[str, str] | None = None,
        n_good: int = 5,
        n_bad: int = 5,
        n_concepts: int = 3,
        max_concepts: int = 20,
    ) -> "LLMContext":
        """Build an LLMContext by abstracting concepts from a PopulationResult.

        Calls the LLM once with the best and worst expressions found during
        training and asks it to hypothesize what mathematical patterns
        distinguish them.

        Parameters
        ----------
        result : PopulationResult
            Fitted result object (one population).
        client : LLMClient
            Configured LLM client.
        description : str
            Natural-language domain description. The richer the better.
        target_name : str or None
            Target variable name. Defaults to the population name.
        variable_names : dict or None
            Mapping from internal names to domain names (e.g. ``{"x1": "burnup"}``).
            If None, uses the feature names already in the population's pset.
        n_good : int
            Number of best HOF expressions to include in the prompt.
        n_bad : int
            Number of worst HOF expressions to include in the prompt.
        n_concepts : int
            Number of concepts to request from the LLM.
        max_concepts : int
            Capacity of the returned context library.

        Returns
        -------
        LLMContext
            Populated with the generated concepts.

        Examples
        --------
        >>> ctx = LLMContext.from_result(
        ...     result["PPF"],
        ...     client=client,
        ...     description="Peak Power Factor in a pressurized water reactor",
        ...     target_name="PPF",
        ... )
        >>> print(ctx.concepts)
        """
        from symgene.llm.concept_abstraction import abstract_concepts

        tname = target_name or result._pop.name
        feat_names = list(result._pop.pset.feature_names)

        good_exprs, bad_exprs = _hof_expressions(result, n_good, n_bad)

        concepts = abstract_concepts(
            good_expressions=good_exprs,
            bad_expressions=bad_exprs,
            client=client,
            description=description,
            target_name=tname,
            feature_names=feat_names,
            n_concepts=n_concepts,
        )

        ctx = cls(
            concepts=concepts,
            variable_names=variable_names or {},
            max_concepts=max_concepts,
            description=description,
            target_name=tname,
        )
        return ctx

    # ------------------------------------------------------------------
    # Evolving the concept library
    # ------------------------------------------------------------------

    def evolve(
        self,
        client: "LLMClient",
        n_concepts: int = 3,
        n_iterations: int = 1,
    ) -> "LLMContext":
        """Refine and extend the concept library using the LLM.

        Calls the LLM ``n_iterations`` times, each time asking it to generate
        new concepts that are refinements or creative combinations of the
        existing ones. All new concepts are added to the library (up to
        ``max_concepts``).

        Parameters
        ----------
        client : LLMClient
            Configured LLM client.
        n_concepts : int
            Number of new concepts to request per iteration.
        n_iterations : int
            Number of evolution rounds to run.

        Returns
        -------
        LLMContext
            Self, for chaining.

        Examples
        --------
        >>> ctx.evolve(client, n_concepts=3, n_iterations=2)
        >>> print(len(ctx))
        """
        from symgene.llm.concept_abstraction import evolve_concepts

        for _ in range(n_iterations):
            if self.is_empty():
                break
            new_concepts = evolve_concepts(
                concepts=self.concepts,
                client=client,
                description=self.description,
                n_concepts=n_concepts,
            )
            self.add_many(new_concepts)

        return self
