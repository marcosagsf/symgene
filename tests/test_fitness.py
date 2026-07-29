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
