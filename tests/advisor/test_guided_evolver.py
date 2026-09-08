import numpy as np
from symgene.population import Population
from symgene.primitive_set import PrimitiveSet
from symgene.advisor import AIGuidedEvolver, StagnationTrigger


def _make_pop(name: str = "target") -> Population:
    pset = PrimitiveSet(n_inputs=2, feature_names=["x1", "x2"])
    pset.add_from_catalog()
    return Population(name=name, pset=pset, n_genes=2, pop_size=10)


_rng = np.random.default_rng(0)
_X = _rng.random((20, 2))
_y = {"target": _X[:, 0] + _X[:, 1]}


def test_aiguidedvolver_completes_without_llm():
    pop = _make_pop()
    guided = AIGuidedEvolver(
        populations=[pop], n_gen=5, llm_client=None, verbose=0,
    )
    result = guided.fit(_X, _y)
    assert "target" in result


def test_result_has_advisor_log_attribute():
    pop = _make_pop()
    guided = AIGuidedEvolver(
        populations=[pop], n_gen=5, llm_client=None, verbose=0,
    )
    result = guided.fit(_X, _y)
    assert hasattr(result, "advisor_log_")
    assert isinstance(result.advisor_log_, list)


def test_no_interventions_without_llm_client():
    pop = _make_pop()
    guided = AIGuidedEvolver(
        populations=[pop], n_gen=5, llm_client=None,
        triggers=[StagnationTrigger(patience=1)], verbose=0,
    )
    result = guided.fit(_X, _y)
    assert result.advisor_log_ == []


def test_result_predictions_have_correct_shape():
    pop = _make_pop()
    guided = AIGuidedEvolver(
        populations=[pop], n_gen=5, llm_client=None, verbose=0,
    )
    result = guided.fit(_X, _y)
    y_pred = result["target"].predict(_X)
    assert y_pred.shape == (20,)


def test_symgeneevolver_result_has_no_advisor_log():
    from symgene.evolver import SymGeneEvolver
    pop = _make_pop()
    evolver = SymGeneEvolver(populations=[pop], n_gen=5, verbose=0)
    result = evolver.fit(_X, _y)
    assert not hasattr(result, "advisor_log_")


def test_default_triggers_applied_when_none_provided():
    pop = _make_pop()
    guided = AIGuidedEvolver(
        populations=[pop], n_gen=3, llm_client=None, verbose=0,
    )
    assert len(guided.triggers) == 3
