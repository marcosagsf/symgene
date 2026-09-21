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


def test_aiguidedvolver_verbose1_does_not_crash(capsys):
    pop = _make_pop()
    guided = AIGuidedEvolver(
        populations=[pop], n_gen=3, llm_client=None, verbose=1,
    )
    guided.fit(_X, _y)
    captured = capsys.readouterr()
    assert "[AIGuidedEvolver]" in captured.out


def test_aiguidedvolver_with_val_data():
    pop = _make_pop()
    X_val = _rng.random((10, 2))
    y_val = {"target": X_val[:, 0] + X_val[:, 1]}
    guided = AIGuidedEvolver(
        populations=[pop], n_gen=5, llm_client=None, verbose=0,
    )
    result = guided.fit(_X, _y, X_val=X_val, y_val=y_val)
    hist = result["target"].history_
    assert "val_r2" in hist[0]


def test_aiguidedvolver_with_seed():
    """Passing seed exercises the seed-setting path in fit()."""
    pop = _make_pop()
    guided = AIGuidedEvolver(
        populations=[pop], n_gen=3, llm_client=None, verbose=0, seed=42,
    )
    result = guided.fit(_X, _y)
    assert "target" in result


def test_aiguidedvolver_with_callbacks():
    """Callbacks are invoked from guided_evolver's fit loop."""
    from symgene.callbacks.base import Callback
    class CountCallback(Callback):
        def __init__(self): self.begin_count = 0
        def on_train_begin(self, logs=None): self.begin_count += 1
    pop = _make_pop()
    cb = CountCallback()
    guided = AIGuidedEvolver(
        populations=[pop], n_gen=3, llm_client=None, verbose=0, callbacks=[cb],
    )
    guided.fit(_X, _y)
    assert cb.begin_count == 1


def test_aiguidedvolver_stagnation_trigger_dry_run(capsys):
    """Trigger fires but no LLM available — dry-run message printed at verbose=1."""
    pop = _make_pop()
    guided = AIGuidedEvolver(
        populations=[pop], n_gen=10, llm_client=None,
        triggers=[StagnationTrigger(patience=1)],
        verbose=1,
    )
    guided.fit(_X, _y)
    captured = capsys.readouterr()
    # dry-run path prints trigger info
    assert len(captured.out) > 0
