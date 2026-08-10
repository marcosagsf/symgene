import random
from functools import partial
from typing import Callable
import deap.gp as gp
from symgene.primitives.squash import Squash
from symgene.primitives.catalog import get_catalog, STANDARD
from symgene.primitives.sympy_map import CATALOG_SYMPY

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

        self.primitives: list[tuple] = []   # (fn, arity, name, sympy_fn | None)
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
            sympy_fn = CATALOG_SYMPY.get(name)
            self.primitives.append((fn, arity, name, sympy_fn))
        return self

    def add_custom(
        self,
        fn: Callable,
        arity: int,
        name: str,
        sympy_fn: Callable | None = None,
    ) -> "PrimitiveSet":
        """Register a custom primitive.

        Parameters
        ----------
        sympy_fn:
            Lambda recebendo argumentos sympy e retornando uma expressão sympy.
            Necessário para que to_sympy() / to_latex() funcionem com esta primitiva.
            Se None, to_sympy() emitirá UserWarning ao encontrar esta primitiva.
        """
        self.primitives.append((fn, arity, name, sympy_fn))
        return self

    def sympy_registry(self) -> dict:
        """Retorna {name: sympy_fn} para todas as primitivas registradas."""
        return {name: sfn for _, _, name, sfn in self.primitives}

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

    @classmethod
    def from_description(
        cls,
        description: str,
        client: object,
        n_inputs: int,
        feature_names: list[str] | None = None,
        available: list[str] | None = None,
        n_min: int = 8,
        n_max: int = 16,
    ) -> "PrimitiveSet":
        """Build a PrimitiveSet whose primitives are chosen by an LLM.

        Requires ``pip install symgene[llm]``.

        Parameters
        ----------
        description : str
            Natural-language description of the modeled phenomenon.
        client : LLMClient
            Configured LLM client from ``symgene.llm``.
        n_inputs : int
            Number of input variables.
        feature_names : list of str or None
            Domain names for the input variables (strongly recommended —
            the LLM uses them to reason about the physics).
        available : list of str or None
            Restrict the catalog exposed to the LLM. Defaults to ALL.
        n_min, n_max : int
            Requested range for the number of primitives to select.

        Returns
        -------
        PrimitiveSet
            Configured instance with LLM-selected primitives already loaded.
            Pass it directly to ``Population(pset=...)``.
            Optionally call ``.add_ephemeral()`` or ``.set_squash()`` before
            passing to Population — the Population calls ``.build()`` internally.

        Raises
        ------
        InsufficientContextError
            If the LLM determines the description is too vague to make
            well-motivated selections. Check ``e.llm_message`` for what
            additional context is needed, then retry with a richer description.

        Examples
        --------
        >>> from symgene.llm import LLMClient, InsufficientContextError
        >>> client = LLMClient(provider="anthropic", model="claude-haiku-4-5-20251001")
        >>> try:
        ...     pset = PrimitiveSet.from_description(
        ...         description="axial power distribution in a PWR with burnup effects",
        ...         client=client,
        ...         n_inputs=5,
        ...         feature_names=["burnup", "enrichment", "boron", "inlet_temp", "power"],
        ...     )
        ... except InsufficientContextError as e:
        ...     print(e.llm_message)  # what the LLM needs to know
        ...
        >>> # Add ephemerals if needed, then plug into Population:
        >>> pset.add_ephemeral("c", low=-1.0, high=1.0)
        >>> pop = Population(name="PPF", pset=pset, ...)
        """
        from symgene.llm.primitive_suggest import suggest_primitives
        names = suggest_primitives(
            description=description,
            client=client,
            feature_names=feature_names,
            available=available,
            n_min=n_min,
            n_max=n_max,
        )
        pset = cls(n_inputs=n_inputs, feature_names=feature_names)
        pset.add_from_catalog(names)
        return pset

    def build(self) -> gp.PrimitiveSet:
        """Compile into a DEAP PrimitiveSet ready for evolution."""
        deap_pset = gp.PrimitiveSet("MAIN", self.n_inputs)

        # rename ARG0..N to feature names
        rename = {f"ARG{i}": name for i, name in enumerate(self.feature_names)}
        deap_pset.renameArguments(**rename)

        # add primitives
        for fn, arity, name, *_ in self.primitives:
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
