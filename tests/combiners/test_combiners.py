import numpy as np
import pytest
from symgene.combiners.ridge import RidgeCombiner
from symgene.combiners.lasso import LassoCombiner
from symgene.combiners.linear import LinearCombiner

rng = np.random.default_rng(42)
G = rng.standard_normal((100, 5))
y = G @ np.array([1, -2, 0.5, 0, 3]) + 0.1 * rng.standard_normal(100)

def test_ridge_fit_predict_shape():
    c = RidgeCombiner(alphas=[1.0, 10.0])
    c.fit(G, y)
    pred = c.predict(G)
    assert pred.shape == (100,)

def test_ridge_coef_shape():
    c = RidgeCombiner(alphas=[1.0])
    c.fit(G, y)
    assert c.coef_.shape == (5,)

def test_ridge_degree2_predict_shape():
    c = RidgeCombiner(alphas=[1.0], degree=2)
    c.fit(G, y)
    pred = c.predict(G)
    assert pred.shape == (100,)

def test_ridge_gene_weights_degree1():
    c = RidgeCombiner(alphas=[1.0], degree=1)
    c.fit(G, y)
    assert c.gene_weights_.shape == (5,)

def test_ridge_gene_weights_degree2():
    c = RidgeCombiner(alphas=[1.0], degree=2)
    c.fit(G, y)
    assert c.gene_weights_.shape == (5,)

def test_lasso_fit_predict():
    c = LassoCombiner(alphas=[0.1, 1.0])
    c.fit(G, y)
    assert c.predict(G).shape == (100,)

def test_linear_fit_predict():
    c = LinearCombiner()
    c.fit(G, y)
    assert c.predict(G).shape == (100,)

def test_ridge_r2_reasonable():
    c = RidgeCombiner(alphas=[1.0])
    c.fit(G, y)
    from sklearn.metrics import r2_score
    assert r2_score(y, c.predict(G)) > 0.95
