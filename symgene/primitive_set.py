import random
from functools import partial
from typing import Callable
import deap.gp as gp
from symgene.primitives.squash import Squash
from symgene.primitives.catalog import get_catalog, STANDARD

class PrimitiveSet:
    def __init__(
        self,
        n_inputs: int,
        feature_names: list[str] | None = None,
    ):
        self.n_inputs = n_inputs
        if feature_names is not None:
            if len(feature_names) != n_inputs:
                raise ValueError(
                    f"feature_names length {len(feature_names)} != n_inputs {n_inputs}"
                )
            self.feature_names = list(feature_names)
        else:
            self.feature_names = [f"x{i+1}" for i in range(n_inputs)]

        self.primitives: list[tuple] = []   # (fn, arity, name)
        self.ephemerals: list[dict] = []
        self.squash: Squash | None = None
        self._custom: list[tuple] = []

    def add_from_catalog(self, names: list[str] | None = None) -> "PrimitiveSet":
        selected = names if names is not None else STANDARD
        sq = self.squash if self.squash is not None else Squash()
        fns = get_catalog(selected, sq)
        import inspect
        for name, fn in fns.items():
            arity = len(inspect.signature(fn).parameters)
            self.primitives.append((fn, arity, name))
        return self

    def add_custom(self, fn: Callable, arity: int, name: str) -> "PrimitiveSet":
        self.primitives.append((fn, arity, name))
        return self

    def add_ephemeral(
        self, name: str, dist: str = "uniform",
        low: float = -1.0, high: float = 1.0,
        mean: float = 0.0, std: float = 1.0, n: int = 1,
    ) -> "PrimitiveSet":
        self.ephemerals.append({
            "name": name, "dist": dist,
            "low": low, "high": high,
            "mean": mean, "std": std, "n": n,
        })
        return self

    def set_squash(
        self,
        lim: float = 8.0,
        alpha: float = 0.1,
        scale: float = 2.0,
        fn: Callable | None = None,
    ) -> "PrimitiveSet":
        self.squash = Squash(lim=lim, alpha=alpha, scale=scale, fn=fn)
        return self

    def disable_squash(self) -> "PrimitiveSet":
        self.squash = None
        return self

    def build(self) -> gp.PrimitiveSet:
        """Compile into a DEAP PrimitiveSet ready for evolution."""
        deap_pset = gp.PrimitiveSet("MAIN", self.n_inputs)

        # rename ARG0..N to feature names
        rename = {f"ARG{i}": name for i, name in enumerate(self.feature_names)}
        deap_pset.renameArguments(**rename)

        # add primitives
        for fn, arity, name in self.primitives:
            deap_pset.addPrimitive(fn, arity, name=name)

        # add ephemerals
        for eph in self.ephemerals:
            dist = eph["dist"]
            for i in range(eph["n"]):
                eph_name = f"{eph['name']}_{i}"
                if dist == "uniform":
                    gen_fn = partial(random.uniform, eph["low"], eph["high"])
                elif dist == "normal":
                    gen_fn = partial(random.gauss, eph["mean"], eph["std"])
                else:
                    raise ValueError(f"Unknown ephemeral dist: {dist}")
                deap_pset.addEphemeralConstant(eph_name, gen_fn)

        return deap_pset
