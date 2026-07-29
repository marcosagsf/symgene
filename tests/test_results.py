import numpy as np
import pytest

rng = np.random.default_rng(0)
X = rng.standard_normal((50, 3))
y = X[:, 0] ** 2 - X[:, 1]

def make_trained_result():
    from symgene.primitive_set import PrimitiveSet
    from symgene.population import Population
    from symgene.evolver import SymGeneEvolver
    from symgene.fitness import FitnessEvaluator
    from symgene.metrics.regression import mse
    pset = PrimitiveSet(n_inputs=3, feature_names=["a", "b", "c"])
    pset.add_from_catalog(["add", "mul", "square", "sin"])
    pset.set_squash(lim=8)
    pop = Population(name="test", pset=pset, n_genes=3, pop_size=10,
                     fitness=FitnessEvaluator(metric=mse), combiner="ridge")
    evolver = SymGeneEvolver(populations=[pop], n_gen=3, seed=0)
    return evolver.fit(X, {"test": y})

def test_result_indexable_by_name():
    results = make_trained_result()
    res = results["test"]
    assert res is not None

def test_result_predict_shape():
    results = make_trained_result()
    y_pred = results["test"].predict(X)
    assert y_pred.shape == (50,)

def test_result_best_individual_not_none():
    results = make_trained_result()
    assert results["test"].best_individual_ is not None

def test_result_best_expression_is_string():
    results = make_trained_result()
    expr = results["test"].best_expression_
    assert isinstance(expr, str)
    assert len(expr) > 0

def test_result_history_is_list():
    results = make_trained_result()
    hist = results["test"].history_
    assert isinstance(hist, list)
    assert len(hist) == 3

def test_result_n_genes():
    results = make_trained_result()
    assert results["test"].n_genes_ >= 1

def test_result_coefficients_shape():
    results = make_trained_result()
    coef = results["test"].coefficients_
    assert coef is not None
    assert len(coef) > 0

def test_result_save_load(tmp_path):
    results = make_trained_result()
    results.save(str(tmp_path / "run1"))
    from symgene.results import SymGeneResult
    loaded = SymGeneResult.load(str(tmp_path / "run1"))
    assert "test" in loaded
