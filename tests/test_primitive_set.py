import pytest
from symgene.primitive_set import PrimitiveSet
from symgene.primitives.catalog import STANDARD

def test_default_feature_names():
    pset = PrimitiveSet(n_inputs=4)
    assert pset.feature_names == ["x1", "x2", "x3", "x4"]

def test_custom_feature_names():
    pset = PrimitiveSet(n_inputs=3, feature_names=["age", "bmd", "weight"])
    assert pset.feature_names == ["age", "bmd", "weight"]

def test_feature_names_length_mismatch_raises():
    with pytest.raises(ValueError):
        PrimitiveSet(n_inputs=3, feature_names=["a", "b"])

def test_add_from_catalog():
    pset = PrimitiveSet(n_inputs=2)
    pset.add_from_catalog(STANDARD)
    names = [p[2] for p in pset.primitives]
    assert "add" in names
    assert "sin" in names

def test_add_from_catalog_by_name_list():
    pset = PrimitiveSet(n_inputs=2)
    pset.add_from_catalog(["sin", "cos"])
    names = [p[2] for p in pset.primitives]
    assert "sin" in names
    assert "cos" in names
    assert "add" not in names

def test_add_custom_primitive():
    pset = PrimitiveSet(n_inputs=3)
    def my_fn(a, b, c): return a + b + c
    pset.add_custom(my_fn, arity=3, name="my_fn")
    names = [p[2] for p in pset.primitives]
    assert "my_fn" in names

def test_add_ephemeral():
    pset = PrimitiveSet(n_inputs=2)
    pset.add_ephemeral("c_small", dist="uniform", low=-1.5, high=1.5, n=3)
    assert len(pset.ephemerals) == 1
    assert pset.ephemerals[0]["name"] == "c_small"
    assert pset.ephemerals[0]["n"] == 3

def test_set_squash():
    pset = PrimitiveSet(n_inputs=2)
    pset.set_squash(lim=5, alpha=0.2, scale=1.0)
    assert pset.squash is not None
    assert pset.squash.lim == 5

def test_disable_squash():
    pset = PrimitiveSet(n_inputs=2)
    pset.set_squash(lim=8)
    pset.disable_squash()
    assert pset.squash is None

def test_build_deap_pset():
    pset = PrimitiveSet(n_inputs=3, feature_names=["a", "b", "c"])
    pset.add_from_catalog(["add", "sin"])
    pset.add_ephemeral("c1", dist="uniform", low=-1.0, high=1.0, n=2)
    deap_pset = pset.build()
    import deap.gp as gp
    assert isinstance(deap_pset, gp.PrimitiveSet)
