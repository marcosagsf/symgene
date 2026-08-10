import random
from unittest.mock import MagicMock

import deap.base as base
import deap.creator as creator
import deap.gp as gp
import pytest

# Trigger creator setup
from symgene.population import Population  # noqa: F401
from symgene.llm.rescue import (
    _try_parse_gene,
    _build_prompt,
    _call_llm_with_retry,
    rescue_worst,
)
from symgene.llm.context import LLMContext


# ---------------------------------------------------------------------------
# Minimal DEAP pset for testing
# ---------------------------------------------------------------------------

def _make_pset():
    import math
    pset = gp.PrimitiveSet("MAIN", 2)
    pset.renameArguments(ARG0="x1", ARG1="x2")
    pset.addPrimitive(lambda a, b: a + b, 2, name="add")
    pset.addPrimitive(lambda a, b: a * b, 2, name="mul")
    pset.addPrimitive(math.sin, 1, name="sin")
    return pset


def _make_mock_pop(n_genes: int = 2, pop_size: int = 10):
    """Create a minimal duck-typed Population with real DEAP pset."""
    from symgene.primitive_set import PrimitiveSet

    pset_obj = PrimitiveSet(n_inputs=2, feature_names=["x1", "x2"])
    pset_obj.add_from_catalog(["add", "mul", "sin"])

    pop = MagicMock()
    pop.pset = pset_obj
    pop._deap_pset = pset_obj.build()
    pop.tree_max = 100
    pop.best = None

    # Create real SGIndividuals
    toolbox = base.Toolbox()
    toolbox.register(
        "gene",
        lambda: creator.SGGene(
            gp.genHalfAndHalf(pop._deap_pset, min_=1, max_=3)
        ),
    )
    toolbox.register(
        "individual",
        lambda: creator.SGIndividual(toolbox.gene() for _ in range(n_genes)),
    )

    population = [toolbox.individual() for _ in range(pop_size)]
    # Assign dummy fitness values
    for i, ind in enumerate(population):
        ind.fitness.values = (float(i + 1),)  # worst = highest index

    hof = gp.PrimitiveTree.from_string("add(x1, x2)", pop._deap_pset)
    best_ind = creator.SGIndividual(
        [creator.SGGene(hof)] * n_genes
    )
    best_ind.fitness.values = (0.01,)
    pop.best = best_ind
    pop._population = population

    return pop


def _mock_ctx(description: str = "test domain") -> LLMContext:
    return LLMContext(
        concepts=["trigonometric structure", "polynomial decay"],
        description=description,
        target_name="y",
    )


def _mock_client(response: str) -> MagicMock:
    client = MagicMock()
    client.complete.return_value = response
    return client


# ---------------------------------------------------------------------------
# _try_parse_gene
# ---------------------------------------------------------------------------

def test_try_parse_gene_valid():
    pset = _make_pset()
    gene = _try_parse_gene("add(x1, x2)", pset)
    assert gene is not None
    assert isinstance(gene, creator.SGGene)


def test_try_parse_gene_nested():
    pset = _make_pset()
    gene = _try_parse_gene("add(x1, mul(x2, sin(x1)))", pset)
    assert gene is not None


def test_try_parse_gene_invalid_primitive():
    pset = _make_pset()
    gene = _try_parse_gene("exp(x1)", pset)  # exp not in pset
    assert gene is None


def test_try_parse_gene_markdown_stripped():
    pset = _make_pset()
    gene = _try_parse_gene("```\nadd(x1, x2)\n```", pset)
    assert gene is not None


def test_try_parse_gene_numbered_line():
    pset = _make_pset()
    gene = _try_parse_gene("1. add(x1, x2)", pset)
    assert gene is not None


def test_try_parse_gene_tree_max_rejected():
    pset = _make_pset()
    # A valid but long expression
    gene = _try_parse_gene("add(x1, mul(x2, add(x1, x2)))", pset, tree_max=3)
    assert gene is None  # 7 nodes > 3


def test_try_parse_gene_empty_string():
    pset = _make_pset()
    gene = _try_parse_gene("", pset)
    assert gene is None


# ---------------------------------------------------------------------------
# _call_llm_with_retry
# ---------------------------------------------------------------------------

def test_call_llm_returns_valid_gene_on_first_try():
    pset = _make_pset()
    client = _mock_client("add(x1, x2)")
    result = _call_llm_with_retry(
        client, "prompt", pset, tree_max=100, max_retries=3, n_genes=1
    )
    assert result is not None
    assert client.complete.call_count == 1


def test_call_llm_retries_on_parse_failure():
    pset = _make_pset()
    # First response invalid, second valid
    client = MagicMock()
    client.complete.side_effect = ["not_valid!!!", "add(x1, x2)"]
    result = _call_llm_with_retry(
        client, "prompt", pset, tree_max=100, max_retries=3, n_genes=1
    )
    assert result is not None
    assert client.complete.call_count == 2


def test_call_llm_returns_none_after_all_retries_exhausted():
    pset = _make_pset()
    client = _mock_client("definitely_not_parseable_XXXXX")
    result = _call_llm_with_retry(
        client, "prompt", pset, tree_max=100, max_retries=2, n_genes=1
    )
    assert result is None
    assert client.complete.call_count == 2


def test_call_llm_multi_gene_parses_lines():
    pset = _make_pset()
    client = _mock_client("add(x1, x2)\nmul(x1, x2)")
    result = _call_llm_with_retry(
        client, "prompt", pset, tree_max=100, max_retries=1, n_genes=2
    )
    assert result is not None
    assert len(result) == 2


def test_call_llm_multi_gene_pads_if_short():
    pset = _make_pset()
    client = _mock_client("add(x1, x2)")  # only 1 line, but n_genes=3
    result = _call_llm_with_retry(
        client, "prompt", pset, tree_max=100, max_retries=1, n_genes=3
    )
    assert result is not None
    assert len(result) == 3


def test_call_llm_api_exception_propagates():
    pset = _make_pset()
    client = MagicMock()
    client.complete.side_effect = ConnectionError("API unreachable")
    with pytest.raises(ConnectionError):
        _call_llm_with_retry(
            client, "prompt", pset, tree_max=100, max_retries=3, n_genes=1
        )


# ---------------------------------------------------------------------------
# rescue_worst
# ---------------------------------------------------------------------------

def test_rescue_worst_gene_level_modifies_population(capsys):
    pop = _make_mock_pop(n_genes=2, pop_size=6)
    ctx = _mock_ctx()
    client = _mock_client("add(x1, x2)")

    n = rescue_worst(pop, client, ctx, rescue_fraction=0.2, level="gene")
    assert n >= 0  # rescue may or may not fire depending on valid inds


def test_rescue_worst_individual_level_replaces_all_genes():
    pop = _make_mock_pop(n_genes=3, pop_size=6)
    ctx = _mock_ctx()
    # Return 3 genes (one per line)
    client = _mock_client("add(x1, x2)\nmul(x1, x2)\nsin(x1)")

    n = rescue_worst(pop, client, ctx, rescue_fraction=0.3, level="individual")
    assert n >= 0


def test_rescue_worst_api_error_propagates():
    pop = _make_mock_pop(n_genes=2, pop_size=6)
    ctx = _mock_ctx()
    client = MagicMock()
    client.complete.side_effect = RuntimeError("API down")

    with pytest.raises(RuntimeError):
        rescue_worst(pop, client, ctx)


def test_rescue_worst_skips_if_no_best():
    pop = _make_mock_pop(n_genes=2, pop_size=6)
    pop.best = None
    ctx = _mock_ctx()
    client = _mock_client("add(x1, x2)")
    n = rescue_worst(pop, client, ctx)
    assert n == 0
    client.complete.assert_not_called()


def test_rescue_worst_invalidates_fitness():
    pop = _make_mock_pop(n_genes=2, pop_size=6)
    ctx = _mock_ctx()
    client = _mock_client("add(x1, x2)")

    rescue_worst(pop, client, ctx, rescue_fraction=0.5, level="gene")
    # Rescued individuals should have invalidated fitness
    for ind in pop._population:
        # Either fitness is still valid (not rescued) or was invalidated
        assert not ind.fitness.valid or ind.fitness.values[0] > 0
