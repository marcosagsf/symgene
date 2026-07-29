import math
from typing import Callable

class Squash:
    """Sigmoidal explosion control for GP tree outputs."""

    def __init__(
        self,
        lim: float = 8.0,
        alpha: float = 0.1,
        scale: float = 2.0,
        fn: Callable[[float], float] | None = None,
        disabled: bool = False,
    ):
        self.lim = lim
        self.alpha = alpha
        self.scale = scale
        self.fn = fn
        self.disabled = disabled

    def __call__(self, x: float) -> float:
        if not math.isfinite(x):
            return 0.0
        if self.disabled:
            return x
        if self.fn is not None:
            return self.fn(x)
        if abs(x) <= self.lim:
            return x
        sign = 1.0 if x > 0 else -1.0
        excess = abs(x) - self.lim
        return sign * (self.lim + self.scale * (1.0 - math.exp(-self.alpha * excess)))
