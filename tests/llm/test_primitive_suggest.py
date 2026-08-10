import json
import warnings
from unittest.mock import MagicMock

import pytest

from symgene.llm.primitive_suggest import suggest_primitives, InsufficientContextError
from symgene.primitive_set import PrimitiveSet


def _mock_client(response: str) -> MagicMock:
    client = MagicMock()
    client.complete.return_value = response
    return client


# ---------------------------------------------------------------------------
# FORMAT A — valid primitive list
# ---------------------------------------------------------------------------

def test_suggest_primitives_valid():
    names = ["add", "mul", "sin", "exp", "log"]
    client = _mock_client(json.dumps(names))
    result = suggest_primitives("test domain", client, feature_names=["x1", "x2"])
    assert result == names


def test_suggest_primitives_filters_unknown():
    payload = ["add", "mul", "NOT_A_PRIMITIVE", "sin"]
    client = _mock_client(json.dumps(payload))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = suggest_primitives("test domain", client)
    assert "NOT_A_PRIMITIVE" not in result
    assert any("NOT_A_PRIMITIVE" in str(warning.message) for warning in w)


def test_suggest_primitives_empty_falls_back_to_standard():
    client = _mock_client(json.dumps(["UNKNOWN_A", "UNKNOWN_B"]))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = suggest_primitives("test domain", client)
    assert len(result) > 0
    assert any("no valid primitives" in str(warning.message).lower() for warning in w)


def test_suggest_primitives_json_embedded_in_text():
    response = 'Sure! Here are my suggestions:\n["add", "sub", "sin"]\nLet me know!'
    client = _mock_client(response)
    result = suggest_primitives("test domain", client)
    assert result == ["add", "sub", "sin"]


# ---------------------------------------------------------------------------
# FORMAT B — needs more information
# ---------------------------------------------------------------------------

def test_suggest_primitives_raises_on_needs_more_info():
    response = json.dumps({
        "needs_more_info": True,
        "message": "Please describe the physical phenomenon being modeled and the expected functional form.",
    })
    client = _mock_client(response)
    with pytest.raises(InsufficientContextError) as exc_info:
        suggest_primitives("model something", client)
    assert "Please describe the physical phenomenon" in exc_info.value.llm_message


def test_insufficient_context_error_has_llm_message():
    response = json.dumps({
        "needs_more_info": True,
        "message": "Specify whether the relationship is periodic, exponential, or polynomial.",
    })
    client = _mock_client(response)
    try:
        suggest_primitives("predict a value", client)
    except InsufficientContextError as e:
        assert e.llm_message == "Specify whether the relationship is periodic, exponential, or polynomial."
        assert "needs more information" in str(e).lower()


def test_needs_more_info_embedded_in_text():
    response = (
        'Based on your description, I cannot make specific choices.\n'
        + json.dumps({"needs_more_info": True, "message": "What physics governs this system?"})
    )
    client = _mock_client(response)
    with pytest.raises(InsufficientContextError):
        suggest_primitives("something vague", client)


# ---------------------------------------------------------------------------
# PrimitiveSet.from_description
# ---------------------------------------------------------------------------

def test_pset_from_description_returns_configured_pset():
    names = ["add", "mul", "sin", "exp"]
    client = _mock_client(json.dumps(names))
    pset = PrimitiveSet.from_description(
        description="heat transfer in a reactor",
        client=client,
        n_inputs=3,
        feature_names=["burnup", "enrichment", "boron"],
    )
    assert isinstance(pset, PrimitiveSet)
    registered = [name for _, _, name, *_ in pset.primitives]
    assert set(names).issubset(set(registered))


def test_pset_from_description_passes_feature_names_to_prompt():
    names = ["add", "mul"]
    client = _mock_client(json.dumps(names))
    PrimitiveSet.from_description(
        description="nuclear power distribution",
        client=client,
        n_inputs=2,
        feature_names=["burnup", "enrichment"],
    )
    call_kwargs = client.complete.call_args
    prompt_text = call_kwargs[1]["prompt"] if call_kwargs[1] else call_kwargs[0][0]
    assert "burnup" in prompt_text
    assert "enrichment" in prompt_text


def test_pset_from_description_propagates_insufficient_context_error():
    response = json.dumps({"needs_more_info": True, "message": "Need more detail."})
    client = _mock_client(response)
    with pytest.raises(InsufficientContextError):
        PrimitiveSet.from_description(
            description="something",
            client=client,
            n_inputs=2,
        )
