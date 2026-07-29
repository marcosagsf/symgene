import math
from symgene.primitives.squash import Squash

def make_arithmetic(squash: Squash) -> dict:
    def add(x, y):   return squash(x + y)
    def sub(x, y):   return squash(x - y)
    def mul(x, y):   return squash(x * y)
    def div(x, y):   return squash(x / y if abs(y) > 1e-9 else 1.0)
    def abs_(x):     return squash(abs(x))
    return {"add": add, "sub": sub, "mul": mul, "div": div, "abs": abs_}
