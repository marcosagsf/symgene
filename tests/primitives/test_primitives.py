import math
import pytest
from symgene.primitives.squash import Squash
from symgene.primitives.arithmetic import make_arithmetic
from symgene.primitives.power import make_power
from symgene.primitives.trigonometric import make_trigonometric
from symgene.primitives.exponential import make_exponential
from symgene.primitives.hyperbolic import make_hyperbolic
from symgene.primitives.activation import make_activation
from symgene.primitives.statistical import make_statistical
from symgene.primitives.logical import make_logical

SQ = Squash(lim=8, alpha=0.1, scale=2.0)

def test_arithmetic_add():
    fns = make_arithmetic(SQ)
    assert fns["add"](2.0, 3.0) == pytest.approx(5.0)

def test_arithmetic_div_zero():
    fns = make_arithmetic(SQ)
    result = fns["div"](1.0, 0.0)
    assert math.isfinite(result)

def test_arithmetic_mul_large():
    fns = make_arithmetic(SQ)
    result = fns["mul"](1000.0, 1000.0)
    assert math.isfinite(result)

def test_power_square():
    fns = make_power(SQ)
    assert fns["square"](2.5) == pytest.approx(6.25)

def test_power_sqrt_negative():
    fns = make_power(SQ)
    result = fns["sqrt"](-4.0)
    assert math.isfinite(result)

def test_power_inv_zero():
    fns = make_power(SQ)
    result = fns["inv"](0.0)
    assert math.isfinite(result)

def test_trig_sin_cos_bounded():
    fns = make_trigonometric(SQ)
    assert abs(fns["sin"](999.0)) <= 8.1
    assert abs(fns["cos"](999.0)) <= 8.1

def test_trig_atan_finite():
    fns = make_trigonometric(SQ)
    assert math.isfinite(fns["atan"](1e10))

def test_exponential_exp_large():
    fns = make_exponential(SQ)
    assert math.isfinite(fns["exp"](1000.0))

def test_exponential_log_negative():
    fns = make_exponential(SQ)
    assert math.isfinite(fns["log"](-5.0))

def test_hyperbolic_tanh_bounded():
    fns = make_hyperbolic(SQ)
    assert abs(fns["tanh"](100.0)) <= 8.1

def test_activation_relu():
    fns = make_activation(SQ)
    assert fns["relu"](-3.0) == pytest.approx(0.0)
    assert fns["relu"](3.0) == pytest.approx(3.0)

def test_activation_sigmoid_bounded():
    fns = make_activation(SQ)
    assert 0.0 <= fns["sigmoid"](0.0) <= 8.1

def test_statistical_mean3():
    fns = make_statistical(SQ)
    assert fns["mean3"](1.0, 2.0, 3.0) == pytest.approx(2.0)

def test_statistical_max3():
    fns = make_statistical(SQ)
    assert fns["max3"](1.0, 5.0, 3.0) == pytest.approx(5.0)

def test_logical_if_positive():
    fns = make_logical(SQ)
    assert fns["if_positive"](1.0, 10.0, -10.0) == pytest.approx(10.0)
    assert fns["if_positive"](-1.0, 10.0, -10.0) == pytest.approx(-10.0)
