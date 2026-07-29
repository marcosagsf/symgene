from symgene.primitives.squash import Squash

def make_logical(squash: Squash) -> dict:
    def if_positive(cond, a, b): return a if cond >= 0 else b
    def if_greater(x, y, a, b): return a if x > y else b
    def step(x): return 1.0 if x >= 0 else 0.0
    return {"if_positive": if_positive, "if_greater": if_greater, "step": step}
