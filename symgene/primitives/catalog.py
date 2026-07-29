from symgene.primitives.squash import Squash
from symgene.primitives.arithmetic import make_arithmetic
from symgene.primitives.power import make_power
from symgene.primitives.trigonometric import make_trigonometric
from symgene.primitives.exponential import make_exponential
from symgene.primitives.hyperbolic import make_hyperbolic
from symgene.primitives.activation import make_activation
from symgene.primitives.statistical import make_statistical
from symgene.primitives.logical import make_logical

def _build_all(squash: Squash) -> dict:
    registry = {}
    for factory in [
        make_arithmetic, make_power, make_trigonometric,
        make_exponential, make_hyperbolic, make_activation,
        make_statistical, make_logical,
    ]:
        registry.update(factory(squash))
    return registry

STANDARD = [
    "add", "sub", "mul", "div", "abs",
    "square", "cube", "sqrt", "inv",
    "sin", "cos", "atan",
    "exp", "log",
    "tanh",
    "sigmoid", "relu", "gaussian",
    "mean2", "mean3", "max2", "max3", "min2", "min3",
]

EXTENDED = STANDARD + [
    "cbrt", "pow",
    "tan", "asin", "acos",
    "log2", "log10",
    "sinh", "cosh", "atanh",
    "softplus", "softsign", "swish", "elu", "sinc",
    "mean4", "max4", "min4", "harmonic_mean2", "geometric_mean2",
    "if_positive", "step",
]

ALL = EXTENDED + ["if_greater"]

_DEFAULT_SQUASH = Squash()

def get_catalog(names: list[str], squash: Squash | None = None) -> dict:
    """Return dict of {name: callable} for the given list of primitive names."""
    sq = squash if squash is not None else _DEFAULT_SQUASH
    registry = _build_all(sq)
    result = {}
    for name in names:
        if name not in registry:
            raise KeyError(f"Unknown primitive: '{name}'. Available: {sorted(registry.keys())}")
        result[name] = registry[name]
    return result
