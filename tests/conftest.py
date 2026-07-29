import numpy as np
import pytest

@pytest.fixture
def rng():
    return np.random.default_rng(42)

@pytest.fixture
def simple_dataset(rng):
    X = rng.uniform(-1, 1, (100, 4))
    y = X[:, 0] ** 2 + X[:, 1] * X[:, 2] - X[:, 3]
    return X, y
