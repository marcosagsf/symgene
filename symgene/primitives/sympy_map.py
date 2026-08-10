"""
Sympy equivalents for every primitive in the symgene catalog.
Used by PopulationResult.to_sympy() to convert DEAP trees to sympy expressions.
The squash wrapper is intentionally omitted — these represent the mathematical intent.

Activation and logical primitives use lazy UndefinedFunction wrappers instead of
sp.Piecewise so that the sympy assumptions system is never triggered on complex
symbolic arguments (which can recurse or hang indefinitely).
"""
import sympy as sp

# Lazy wrappers — never auto-evaluate, safe for arbitrarily deep symbolic args
_relu     = sp.Function("relu")
_sigmoid  = sp.Function("sigmoid")
_softplus = sp.Function("softplus")
_softsign = sp.Function("softsign")
_swish    = sp.Function("swish")
_elu      = sp.Function("elu")
_sinc     = sp.Function("sinc")
_step     = sp.Function("step")
_ifpos    = sp.Function("if_positive")
_ifgt     = sp.Function("if_greater")

CATALOG_SYMPY: dict = {
    # ── Arithmetic ──────────────────────────────────────────────────────────
    "add":      lambda x, y: x + y,
    "sub":      lambda x, y: x - y,
    "mul":      lambda x, y: x * y,
    "div":      lambda x, y: x / y,
    "abs":      lambda x: sp.Abs(x),
    # ── Power ────────────────────────────────────────────────────────────────
    "square":   lambda x: x ** 2,
    "cube":     lambda x: x ** 3,
    "sqrt":     lambda x: sp.sqrt(sp.Abs(x)),
    "cbrt":     lambda x: sp.cbrt(x),
    "inv":      lambda x: 1 / x,
    "pow":      lambda x, y: sp.Abs(x) ** sp.Abs(y),
    # ── Trigonometric ────────────────────────────────────────────────────────
    "sin":      lambda x: sp.sin(x),
    "cos":      lambda x: sp.cos(x),
    "tan":      lambda x: sp.tan(x),
    "atan":     lambda x: sp.atan(x),
    "asin":     lambda x: sp.asin(x),
    "acos":     lambda x: sp.acos(x),
    # ── Exponential / Logarithm ──────────────────────────────────────────────
    "exp":      lambda x: sp.exp(x),
    "log":      lambda x: sp.log(sp.Abs(x)),
    "log2":     lambda x: sp.log(sp.Abs(x), 2),
    "log10":    lambda x: sp.log(sp.Abs(x), 10),
    # ── Hyperbolic ───────────────────────────────────────────────────────────
    "tanh":     lambda x: sp.tanh(x),
    "sinh":     lambda x: sp.sinh(x),
    "cosh":     lambda x: sp.cosh(x),
    "atanh":    lambda x: sp.atanh(x),
    # ── Activation — lazy wrappers avoid Piecewise/assumptions hang ──────────
    "sigmoid":  _sigmoid,
    "relu":     _relu,
    "gaussian": lambda x: sp.exp(-(x ** 2)),
    "softplus": _softplus,
    "softsign": _softsign,
    "swish":    _swish,
    "elu":      _elu,
    "sinc":     _sinc,
    # ── Statistical ─────────────────────────────────────────────────────────
    "mean2":            lambda x, y: (x + y) / 2,
    "mean3":            lambda x, y, z: (x + y + z) / 3,
    "mean4":            lambda x, y, z, w: (x + y + z + w) / 4,
    "max2":             lambda x, y: sp.Max(x, y),
    "max3":             lambda x, y, z: sp.Max(x, y, z),
    "max4":             lambda x, y, z, w: sp.Max(x, y, z, w),
    "min2":             lambda x, y: sp.Min(x, y),
    "min3":             lambda x, y, z: sp.Min(x, y, z),
    "min4":             lambda x, y, z, w: sp.Min(x, y, z, w),
    "harmonic_mean2":   lambda x, y: 2 * x * y / (x + y),
    "geometric_mean2":  lambda x, y: sp.sqrt(sp.Abs(x * y)),
    # ── Logical — lazy wrappers to avoid Piecewise/assumptions hang ──────────
    "if_positive":  lambda x, y, z: _ifpos(x, y, z),
    "step":         _step,
    "if_greater":   lambda x, y, z, w: _ifgt(x, y, z, w),
}
