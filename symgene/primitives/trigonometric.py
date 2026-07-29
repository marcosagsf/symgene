import math
from symgene.primitives.squash import Squash

def make_trigonometric(squash: Squash) -> dict:
    def sin(x):    return squash(math.sin(squash(x)))
    def cos(x):    return squash(math.cos(squash(x)))
    def tan(x):
        v = squash(x)
        try: return squash(math.tan(v))
        except: return 0.0
    def atan(x):   return squash(math.atan(squash(x)))
    def asin(x):   return squash(math.asin(max(-1.0, min(1.0, squash(x) / max(1.0, abs(squash(x)))))))
    def acos(x):   return squash(math.acos(max(-1.0, min(1.0, squash(x) / max(1.0, abs(squash(x)))))))
    return {"sin": sin, "cos": cos, "tan": tan, "atan": atan, "asin": asin, "acos": acos}
