import math
from symgene.primitives.squash import Squash

def make_exponential(squash: Squash) -> dict:
    def exp(x):
        try: return squash(math.exp(min(squash(x), 700.0)))
        except: return squash(1e300)
    def log(x):   return squash(math.log(abs(x)) if abs(x) > 1e-9 else -20.0)
    def log2(x):  return squash(math.log2(abs(x)) if abs(x) > 1e-9 else -66.0)
    def log10(x): return squash(math.log10(abs(x)) if abs(x) > 1e-9 else -20.0)
    return {"exp": exp, "log": log, "log2": log2, "log10": log10}
