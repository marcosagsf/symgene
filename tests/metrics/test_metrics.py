import numpy as np
import pytest
from symgene.metrics.regression import mse, rmse, mae, r2, nrmse, mape

y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])

def test_mse():
    assert mse(y_true, y_pred) == pytest.approx(0.022, rel=0.01)

def test_rmse():
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(mse(y_true, y_pred)))

def test_mae():
    assert mae(y_true, y_pred) == pytest.approx(0.14, rel=0.01)

def test_r2_perfect():
    assert r2(y_true, y_true) == pytest.approx(1.0)

def test_r2_reasonable():
    assert r2(y_true, y_pred) > 0.98

def test_nrmse_positive():
    assert nrmse(y_true, y_pred) > 0

def test_mape_positive():
    assert mape(y_true, y_pred) > 0

# complexity metrics
from symgene.metrics.complexity import n_nodes, complexity_penalty, missing_vars_penalty

class FakeGene:
    def __len__(self): return 10
    def __str__(self): return "add(x1, mul(x2, x3))"

class FakeIndividual(list):
    pass

def make_individual(genes):
    ind = FakeIndividual(genes)
    return ind

def test_n_nodes():
    g1, g2 = FakeGene(), FakeGene()
    ind = make_individual([g1, g2])
    assert n_nodes(ind) == 20

def test_complexity_penalty_zero_lambda():
    g = FakeGene()
    pen = complexity_penalty(lambda_=0.0, tree_max_ref=80)
    ind = make_individual([g])
    assert pen(ind, None, None) == pytest.approx(0.0)

def test_complexity_penalty_positive():
    pen = complexity_penalty(lambda_=1e-3, tree_max_ref=80)
    ind = make_individual([FakeGene(), FakeGene()])
    assert pen(ind, None, None) > 0.0

def test_missing_vars_penalty_none_missing():
    required = {"x1", "x2", "x3"}
    pen = missing_vars_penalty(required=required, beta=0.01)
    g = FakeGene()
    ind = make_individual([g])
    assert pen(ind, None, None) == pytest.approx(0.0)

def test_missing_vars_penalty_all_missing():
    required = {"x1", "x2", "x3"}
    pen = missing_vars_penalty(required=required, beta=1.0)

    class EmptyGene:
        def __len__(self): return 5
        def __str__(self): return "add(x99, x100)"

    ind = make_individual([EmptyGene()])
    assert pen(ind, None, None) > 0.0
