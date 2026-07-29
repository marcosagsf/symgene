from symgene.primitives.catalog import STANDARD, EXTENDED, ALL, get_catalog
from symgene.primitives.squash import Squash

SQ = Squash(lim=8, alpha=0.1, scale=2.0)

def test_standard_has_required_keys():
    fns = get_catalog(STANDARD, SQ)
    for name in ["add", "sub", "mul", "div", "square", "sqrt", "sin", "cos", "tanh", "sigmoid", "relu"]:
        assert name in fns, f"'{name}' missing from STANDARD"

def test_extended_is_superset_of_standard():
    std = set(get_catalog(STANDARD, SQ).keys())
    ext = set(get_catalog(EXTENDED, SQ).keys())
    assert std.issubset(ext)

def test_all_is_superset_of_extended():
    ext = set(get_catalog(EXTENDED, SQ).keys())
    all_ = set(get_catalog(ALL, SQ).keys())
    assert ext.issubset(all_)

def test_get_catalog_by_name_list():
    fns = get_catalog(["sin", "cos", "add"], SQ)
    assert set(fns.keys()) == {"sin", "cos", "add"}

def test_get_catalog_unknown_name_raises():
    import pytest
    with pytest.raises(KeyError):
        get_catalog(["nonexistent_fn"], SQ)

def test_all_functions_return_finite():
    import math
    import inspect
    fns = get_catalog(ALL, SQ)
    test_inputs = {
        1: [1.0],
        2: [1.0, 2.0],
        3: [1.0, 2.0, 3.0],
        4: [1.0, 2.0, 3.0, 4.0],
    }
    for name, fn in fns.items():
        n = len(inspect.signature(fn).parameters)
        args = test_inputs.get(n, [1.0] * n)
        result = fn(*args)
        assert math.isfinite(result), f"{name}({args}) returned {result}"
