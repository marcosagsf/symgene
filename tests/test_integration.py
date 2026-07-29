import numpy as np
from symgene import PrimitiveSet, Population, SymGeneEvolver, SymGeneRegressor
from symgene.primitives import STANDARD
from symgene.fitness import FitnessEvaluator
from symgene.metrics.regression import mse, r2
from symgene.metrics.complexity import complexity_penalty

rng = np.random.default_rng(42)
X = rng.standard_normal((100, 5))
y = X[:, 0] ** 2 + np.sin(X[:, 1]) - X[:, 2] * X[:, 3] + 0.1 * rng.standard_normal(100)

def test_full_pipeline_single_output():
    """End-to-end: PrimitiveSet → Population → Evolver → Result → Predict"""
    pset = PrimitiveSet(n_inputs=5, feature_names=[f"f{i}" for i in range(5)])
    pset.add_from_catalog(STANDARD)
    pset.set_squash(lim=8, alpha=0.1, scale=2.0)

    evaluator = FitnessEvaluator(
        metric=mse,
        penalties=[complexity_penalty(lambda_=5e-4, tree_max_ref=80)],
    )

    pop = Population(
        name="target", pset=pset, n_genes=4, pop_size=20,
        combiner="ridge", ridge_alphas=[1.0, 5.0],
        regression_degree=1, fitness=evaluator,
    )

    evolver = SymGeneEvolver(populations=[pop], n_gen=5, seed=0, verbose=0)
    results = evolver.fit(X, {"target": y})

    res = results["target"]
    y_pred = res.predict(X)
    assert y_pred.shape == (100,)
    assert res.best_expression_ != ""
    assert len(res.history_) == 5
    assert res.n_genes_ >= 1
    assert res.pareto_front_ is not None

def test_regressor_convenience():
    model = SymGeneRegressor(
        n_genes=4, pop_size=20, n_gen=5,
        primitives=STANDARD, seed=0, verbose=0,
    )
    model.fit(X, y)
    y_pred = model.predict(X)
    assert y_pred.shape == (100,)
    assert isinstance(model.best_expression_, str)
