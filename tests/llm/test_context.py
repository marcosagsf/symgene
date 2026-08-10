import json
from unittest.mock import MagicMock

import pytest

from symgene.llm.context import LLMContext
from symgene.llm.concept_abstraction import abstract_concepts, evolve_concepts


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _mock_client(response: str) -> MagicMock:
    client = MagicMock()
    client.complete.return_value = response
    return client


def _mock_pop_result(hof_size: int = 6, name: str = "PPF") -> MagicMock:
    """Create a duck-typed PopulationResult with a fake HOF."""
    mock_gene = MagicMock()
    mock_gene.__str__ = MagicMock(return_value="add(x1, mul(x2, x3))")

    mock_ind = MagicMock()
    mock_ind.__iter__ = MagicMock(return_value=iter([mock_gene]))

    mock_hof = [mock_ind] * hof_size

    mock_pop = MagicMock()
    mock_pop.name = name
    mock_pop.pset.feature_names = ["burnup", "enrichment", "boron"]
    mock_pop._hof = mock_hof
    mock_pop.best = mock_ind

    result = MagicMock()
    result._pop = mock_pop
    result.best_individual_ = mock_ind
    return result


# ---------------------------------------------------------------------------
# LLMContext unit tests
# ---------------------------------------------------------------------------

def test_context_add_respects_max_concepts():
    ctx = LLMContext(max_concepts=3)
    for i in range(5):
        ctx.add(f"concept {i}")
    assert len(ctx) == 3
    assert ctx.concepts[-1] == "concept 4"


def test_context_sample_returns_subset():
    ctx = LLMContext(concepts=["a", "b", "c", "d", "e"])
    sample = ctx.sample(3)
    assert len(sample) == 3
    assert all(c in ctx.concepts for c in sample)


def test_context_sample_fewer_than_n():
    ctx = LLMContext(concepts=["a", "b"])
    sample = ctx.sample(10)
    assert len(sample) == 2


def test_context_is_empty():
    ctx = LLMContext()
    assert ctx.is_empty()
    ctx.add("something")
    assert not ctx.is_empty()


def test_context_add_many():
    ctx = LLMContext(max_concepts=5)
    ctx.add_many(["a", "b", "c"])
    assert len(ctx) == 3


def test_context_repr():
    ctx = LLMContext(concepts=["c1", "c2"], target_name="PPF")
    r = repr(ctx)
    assert "n_concepts=2" in r
    assert "PPF" in r


# ---------------------------------------------------------------------------
# abstract_concepts
# ---------------------------------------------------------------------------

def test_abstract_concepts_returns_list():
    concepts = ["trigonometric structure", "exponential decay"]
    client = _mock_client(json.dumps(concepts))
    result = abstract_concepts(
        good_expressions=["sin(x1)", "exp(x2)"],
        bad_expressions=["add(x1, x2)"],
        client=client,
        description="nuclear flux distribution",
        n_concepts=2,
    )
    assert result == concepts


def test_abstract_concepts_passes_domain_to_prompt():
    client = _mock_client(json.dumps(["some concept"]))
    abstract_concepts(
        good_expressions=["sin(x1)"],
        bad_expressions=["x1"],
        client=client,
        description="critical boron concentration in a PWR",
        target_name="CBORON",
        feature_names=["burnup", "enrichment"],
    )
    call_kwargs = client.complete.call_args
    prompt = call_kwargs[1].get("prompt") or call_kwargs[0][0]
    assert "critical boron concentration" in prompt
    assert "burnup" in prompt


def test_abstract_concepts_returns_empty_on_bad_response():
    client = _mock_client("I cannot identify any patterns.")
    result = abstract_concepts(
        good_expressions=["sin(x1)"],
        bad_expressions=["x1"],
        client=client,
    )
    assert result == []


def test_abstract_concepts_handles_empty_hof():
    client = _mock_client(json.dumps(["periodic structure"]))
    result = abstract_concepts(
        good_expressions=[],
        bad_expressions=[],
        client=client,
        description="test",
    )
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# evolve_concepts
# ---------------------------------------------------------------------------

def test_evolve_concepts_returns_list():
    new_concepts = ["refined: sinusoidal burnup coupling"]
    client = _mock_client(json.dumps(new_concepts))
    result = evolve_concepts(
        concepts=["trigonometric structure", "exponential decay"],
        client=client,
        description="nuclear flux",
    )
    assert result == new_concepts


def test_evolve_concepts_passes_existing_to_prompt():
    client = _mock_client(json.dumps(["new concept"]))
    evolve_concepts(
        concepts=["periodic structure", "ratio of variables"],
        client=client,
        description="heat transfer",
    )
    call_kwargs = client.complete.call_args
    prompt = call_kwargs[1].get("prompt") or call_kwargs[0][0]
    assert "periodic structure" in prompt
    assert "ratio of variables" in prompt


def test_evolve_concepts_empty_input_returns_empty():
    client = _mock_client(json.dumps(["something"]))
    result = evolve_concepts(concepts=[], client=client)
    assert result == []


# ---------------------------------------------------------------------------
# LLMContext.from_result
# ---------------------------------------------------------------------------

def test_from_result_populates_concepts():
    concepts = ["sinusoidal structure", "exponential burnup term"]
    client = _mock_client(json.dumps(concepts))
    pop_result = _mock_pop_result(hof_size=6)

    ctx = LLMContext.from_result(
        result=pop_result,
        client=client,
        description="neutron flux distribution",
        target_name="flux",
    )
    assert isinstance(ctx, LLMContext)
    assert ctx.concepts == concepts
    assert ctx.description == "neutron flux distribution"
    assert ctx.target_name == "flux"


def test_from_result_uses_population_name_as_default_target():
    client = _mock_client(json.dumps(["concept"]))
    pop_result = _mock_pop_result(name="CBORON")

    ctx = LLMContext.from_result(result=pop_result, client=client)
    assert ctx.target_name == "CBORON"


def test_from_result_handles_empty_hof():
    client = _mock_client(json.dumps(["some concept"]))
    pop_result = _mock_pop_result(hof_size=0)

    ctx = LLMContext.from_result(result=pop_result, client=client)
    assert isinstance(ctx, LLMContext)


# ---------------------------------------------------------------------------
# LLMContext.evolve
# ---------------------------------------------------------------------------

def test_context_evolve_adds_concepts():
    initial = ["trigonometric", "polynomial"]
    new_concepts = ["sinusoidal burnup coupling", "quadratic enrichment term"]
    client = _mock_client(json.dumps(new_concepts))

    ctx = LLMContext(concepts=list(initial))
    ctx.evolve(client, n_concepts=2, n_iterations=1)

    assert len(ctx) == len(initial) + len(new_concepts)
    for c in new_concepts:
        assert c in ctx.concepts


def test_context_evolve_multiple_iterations():
    client = _mock_client(json.dumps(["new concept"]))
    ctx = LLMContext(concepts=["seed concept"])
    ctx.evolve(client, n_concepts=1, n_iterations=3)

    assert client.complete.call_count == 3


def test_context_evolve_skips_if_empty():
    client = _mock_client(json.dumps(["new"]))
    ctx = LLMContext()
    ctx.evolve(client, n_iterations=3)
    client.complete.assert_not_called()


def test_context_evolve_returns_self_for_chaining():
    client = _mock_client(json.dumps(["c"]))
    ctx = LLMContext(concepts=["seed"])
    returned = ctx.evolve(client)
    assert returned is ctx
