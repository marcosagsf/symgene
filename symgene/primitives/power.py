import math
from symgene.primitives.squash import Squash

def make_power(squash: Squash) -> dict:
    def square(x):    return squash(x * x)
    def cube(x):      return squash(x * x * x)
    def sqrt(x):      return squash(math.sqrt(abs(x)))
    def cbrt(x):      sign = 1 if x >= 0 else -1; return squash(sign * abs(x) ** (1/3))
    def inv(x):       return squash(1.0 / x if abs(x) > 1e-9 else 1.0)
    def pow2(x, y):   return squash(abs(x) ** min(abs(y), 5.0) if x != 0 else 0.0)
    return {"square": square, "cube": cube, "sqrt": sqrt, "cbrt": cbrt, "inv": inv, "pow": pow2}
