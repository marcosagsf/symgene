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

def test_to_callable_matches_predict():
    """to_callable() must return the same values as predict() for 10 points."""
    results = make_trained_result()
    pop_result = results["test"]
    X_test = np.random.default_rng(99).standard_normal((10, 3))
    fn = pop_result.to_callable()
    assert fn is not None
    np.testing.assert_allclose(fn(X_test), pop_result.predict(X_test), rtol=1e-10)

def test_to_sympy_warns_when_no_sympy_fn():
    """to_sympy() emits UserWarning when a primitive lacks sympy_fn."""
    import warnings
    from symgene import PrimitiveSet, Population, SymGeneEvolver, FitnessEvaluator
    from symgene.metrics.regression import mse as _mse

    rng2 = np.random.default_rng(1)
    X2 = rng2.standard_normal((30, 2))
    y2 = X2[:, 0] ** 2

    pset = PrimitiveSet(n_inputs=2)
    pset.add_custom(lambda a, b: a + b, arity=2, name="myadd")   # no sympy_fn
    pset.add_custom(lambda a: a * a, arity=1, name="mysquare")   # no sympy_fn
    pset.set_squash(lim=8)

    pop = Population(name="p1", pset=pset, n_genes=2, pop_size=10,
                     fitness=FitnessEvaluator(metric=_mse), combiner="ridge")
    evolver = SymGeneEvolver(populations=[pop], n_gen=3, seed=0, verbose=0)
    result = evolver.fit(X2, {"p1": y2})

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result["p1"].to_sympy()

    user_warns = [x for x in w if issubclass(x.category, UserWarning)]
    assert len(user_warns) > 0
    assert "sympy_fn" in str(user_warns[0].message)

def test_pareto_front_property():
    results = make_trained_result()
    pf = results["test"].pareto_front_
    assert isinstance(pf, list)

def test_best_by_accuracy():
    results = make_trained_result()
    ind = results["test"].best_by_accuracy()
    assert ind is not None

def test_best_by_simplicity():
    results = make_trained_result()
    ind = results["test"].best_by_simplicity(max_error=1e9)
    assert ind is not None

def test_best_by_pareto():
    results = make_trained_result()
    ind = results["test"].best_by_pareto(weight=0.5)
    assert ind is not None

def test_plot_convergence_does_not_crash():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    results = make_trained_result()
    fig, ax = plt.subplots()
    results["test"].plot_convergence(ax=ax)
    plt.close(fig)

def test_plot_prediction_does_not_crash():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    results = make_trained_result()
    fig, ax = plt.subplots()
    results["test"].plot_prediction(X, y, ax=ax)
    plt.close(fig)

def test_plot_pareto_does_not_crash():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    results = make_trained_result()
    fig, ax = plt.subplots()
    results["test"].plot_pareto(X, y, ax=ax)
    plt.close(fig)

def test_plot_gene_weights_does_not_crash():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    results = make_trained_result()
    fig, ax = plt.subplots()
    results["test"].plot_gene_weights(ax=ax)
    plt.close(fig)

def test_plot_expression_tree_does_not_crash():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    results = make_trained_result()
    fig, ax = plt.subplots()
    results["test"].plot_expression_tree(gene_idx=0, ax=ax)
    plt.close(fig)

def test_result_save_load(tmp_path):
    results = make_trained_result()
    results.save(str(tmp_path / "run1"))
    from symgene.results import SymGeneResult
    loaded = SymGeneResult.load(str(tmp_path / "run1"))
    assert "test" in loaded
