import math
import pytest
from symgene.primitives.squash import Squash

def test_squash_identity_within_limit():
    s = Squash(lim=8, alpha=0.1, scale=2.0)
    assert s(0.0) == pytest.approx(0.0)
    assert s(4.0) == pytest.approx(4.0)
    assert s(-4.0) == pytest.approx(-4.0)
    assert s(8.0) == pytest.approx(8.0)

def test_squash_compresses_above_limit():
    s = Squash(lim=8, alpha=0.1, scale=2.0)
    assert s(100.0) < 10.5   # bounded, never explodes
    assert s(-100.0) > -10.5

def test_squash_monotone():
    s = Squash(lim=8, alpha=0.1, scale=2.0)
    vals = [s(float(x)) for x in range(-20, 21)]
    assert vals == sorted(vals)

def test_squash_disabled_passes_through():
    s = Squash(disabled=True)
    assert s(999.0) == 999.0

def test_squash_custom_fn():
    fn = lambda x: max(-5.0, min(5.0, x))
    s = Squash(fn=fn)
    assert s(100.0) == pytest.approx(5.0)
    assert s(-100.0) == pytest.approx(-5.0)

def test_squash_nan_returns_zero():
    s = Squash(lim=8, alpha=0.1, scale=2.0)
    assert s(float('nan')) == pytest.approx(0.0)

def test_squash_inf_compresses():
    s = Squash(lim=8, alpha=0.1, scale=2.0)
    result = s(float('inf'))
    assert math.isfinite(result)
