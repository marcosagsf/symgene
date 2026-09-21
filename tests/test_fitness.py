import numpy as np
import pytest
from symgene.fitness import FitnessEvaluator
from symgene.metrics.regression import mse
from symgene.metrics.complexity import complexity_penalty

rng = np.random.default_rng(0)
X = rng.standard_normal((50, 3))
y = X[:, 0] + X[:, 1] ** 2

def test_evaluator_metric_only():
    ev = FitnessEvaluator(metric=mse)
    assert ev.metric is mse
    assert ev.penalties == []

def test_evaluator_with_penalty():
    ev = FitnessEvaluator(metric=mse, penalties=[complexity_penalty(1e-3)])
    assert len(ev.penalties) == 1

def test_evaluator_compute_basic():
    ev = FitnessEvaluator(metric=mse)

    class FakeInd:
        def __iter__(self): return iter([])
        def __len__(self): return 0

    score = ev.compute(FakeInd(), X, y, y_pred=y + 0.1)
    assert score == pytest.approx(mse(y, y + 0.1))

def test_evaluator_compute_with_penalty():
    pen_called = []
    def my_pen(ind, X, y): pen_called.append(True); return 0.5

    ev = FitnessEvaluator(metric=mse, penalties=[my_pen])

    class FakeInd:
        def __iter__(self): return iter([])
        def __len__(self): return 0

    ev.compute(FakeInd(), X, y, y_pred=y)
    assert len(pen_called) == 1

def test_evaluator_custom_penalty_fn():
    custom = lambda ind, X, y: 99.0
    ev = FitnessEvaluator(metric=mse, penalties=[custom])

    class FakeInd:
        def __iter__(self): return iter([])
        def __len__(self): return 0

    score = ev.compute(FakeInd(), X, y, y_pred=y)
    assert score == pytest.approx(0.0 + 99.0)

def test_evaluator_complexity_and_missing_vars_combined():
    """Both penalties accumulate correctly when applied together."""
    from symgene.metrics.complexity import complexity_penalty, missing_vars_penalty

    class FakeGene:
        def __str__(self): return "mul(x1, x1)"
        def __len__(self): return 3
        def __iter__(self): return iter([])

    class FakeInd:
        def __iter__(self): return iter([FakeGene()])
        def __len__(self): return 1

    ind = FakeInd()
    cplex = complexity_penalty(lambda_=1e-3, tree_max_ref=80)
    missing = missing_vars_penalty(required={"x1", "x2"}, beta=0.01)
    ev = FitnessEvaluator(metric=mse, penalties=[cplex, missing])

    score = ev.compute(ind, X, y, y_pred=y)  # MSE = 0

    # complexity: 3 nodes / max(1, 1*80) * 1e-3 = 3.75e-5
    # missing: x1 found, x2 missing → 1/2 * 0.01 = 0.005
    expected = 1e-3 * (3 / max(1, 1 * 80)) + 0.01 * (1 / 2)
    assert score == pytest.approx(expected)
