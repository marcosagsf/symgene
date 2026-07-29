import math
from symgene.primitives.squash import Squash

def make_hyperbolic(squash: Squash) -> dict:
    def tanh(x):   return squash(math.tanh(squash(x)))
    def sinh(x):
        v = squash(x)
        try: return squash(math.sinh(v))
        except: return squash(1e300 if v > 0 else -1e300)
    def cosh(x):
        v = squash(x)
        try: return squash(math.cosh(v))
        except: return squash(1e300)
    def atanh(x):
        v = max(-0.9999, min(0.9999, squash(x) / max(1.0, abs(squash(x)) + 1.0)))
        return squash(math.atanh(v))
    return {"tanh": tanh, "sinh": sinh, "cosh": cosh, "atanh": atanh}
