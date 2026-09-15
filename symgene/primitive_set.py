import inspect
import random
from functools import partial
from typing import Any, Callable
import deap.gp as gp
from symgene.primitives.squash import Squash
from symgene.primitives.catalog import get_catalog, STANDARD
from symgene.primitives.sympy_map import CATALOG_SYMPY

class PrimitiveSet:
    """Registry of mathematical primitives and terminal symbols for MGGP.

    Builds the search space for symbolic regression by collecting primitive
    functions (operators), ephemeral constants (random terminals), and an
    optional output squash function. Call :meth:`build` to compile into a
    DEAP ``PrimitiveSet`` ready for evolution.

    Parameters
    ----------
    n_inputs : int
        Number of input variables (features).
    feature_names : list of str, optional
        Human-readable names for each input variable. Length must equal
        ``n_inputs``. Defaults to ``["x1", "x2", ...]``.

    Raises
    ------
    ValueError
        If ``feature_names`` is provided but its length differs from
        ``n_inputs``.

    Examples
    --------
    >>> from symgene import PrimitiveSet
    >>> pset = PrimitiveSet(n_inputs=2, feature_names=["Re", "Pr"])
    >>> pset.add_from_catalog(["add", "mul", "sin", "cos"])
    PrimitiveSet(n_inputs=2, primitives=4)
    >>> deap_pset = pset.build()
    >>> deap_pset.arity
    2
    """

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

    def __repr__(self) -> str:
        return f"PrimitiveSet(n_inputs={self.n_inputs}, primitives={len(self.primitives)})"

    def add_from_catalog(self, names: list[str] | None = None) -> "PrimitiveSet":
        """Load primitives from the built-in catalog.

        Parameters
        ----------
        names : list of str, optional
            Catalog keys to load (e.g. ``["add", "mul", "sin"]``). Defaults
            to the ``STANDARD`` preset defined in
            ``symgene.primitives.catalog``.

        Returns
        -------
        PrimitiveSet
            ``self``, for method chaining.
        """
        selected = names if names is not None else STANDARD
        sq = self.squash if self.squash is not None else Squash()
        fns = get_catalog(selected, sq)
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
        """Register a user-defined primitive function.

        Parameters
        ----------
        fn : Callable
            The primitive function. Must accept exactly ``arity`` positional
            float arguments and return a float.
        arity : int
            Number of arguments that ``fn`` expects.
        name : str
            Unique string identifier used in evolved expressions.
        sympy_fn : Callable or None, optional
            Lambda that accepts ``arity`` SymPy arguments and returns a SymPy
            expression. Required for :meth:`PopulationResult.to_sympy` and
            :meth:`PopulationResult.to_latex` to work with this primitive.
            If ``None``, those methods will emit a ``UserWarning`` when the
            primitive appears in the best individual.

        Returns
        -------
        PrimitiveSet
            ``self``, for method chaining.

        Examples
        --------
        >>> from symgene import PrimitiveSet
        >>> pset = PrimitiveSet(n_inputs=1, feature_names=["x"])
        >>> pset.add_custom(
        ...     fn=lambda x: x ** 3,
        ...     arity=1,
        ...     name="cube",
        ...     sympy_fn=lambda x: x ** 3,
        ... )
        PrimitiveSet(n_inputs=1, primitives=1)
        """
        self.primitives.append((fn, arity, name, sympy_fn))
        return self

    def sympy_registry(self) -> dict[str, Any]:
        """Return ``{name: sympy_fn}`` for all registered primitives."""
        return {name: sfn for _, _, name, sfn in self.primitives}

    def add_ephemeral(
        self, name: str, dist: str = "uniform",
        low: float = -1.0, high: float = 1.0,
        mean: float = 0.0, std: float = 1.0, n: int = 1,
    ) -> "PrimitiveSet":
        """Add random constant terminal(s) sampled at tree-generation time.

        Each ephemeral becomes a leaf node whose value is drawn from the
        specified distribution every time a new tree is created.

        Parameters
        ----------
        name : str
            Base name for the ephemeral. If ``n > 1``, terminals are named
            ``name_0``, ``name_1``, … ``name_{n-1}``.
        dist : {"uniform", "normal"}
            Sampling distribution.
        low : float
            Lower bound (uniform only). Default ``-1.0``.
        high : float
            Upper bound (uniform only). Default ``1.0``.
        mean : float
            Mean (normal only). Default ``0.0``.
        std : float
            Standard deviation (normal only). Default ``1.0``.
        n : int
            Number of independent ephemeral constants to register.
            Default ``1``.

        Returns
        -------
        PrimitiveSet
            ``self``, for method chaining.

        Raises
        ------
        ValueError
            If ``dist`` is not ``"uniform"`` or ``"normal"``.
        """
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
        """Configure the output squashing function applied to gene outputs.

        Squashing bounds extreme values before they reach the combiner,
        preventing numerical overflow during evolution.

        Parameters
        ----------
        lim : float
            Symmetric soft-clamp limit. Default ``8.0``.
        alpha : float
            Slope beyond the limit (leaky region). Default ``0.1``.
        scale : float
            Overall scaling factor. Default ``2.0``.
        fn : Callable or None
            Custom squash function overriding the built-in soft-clamp.

        Returns
        -------
        PrimitiveSet
            ``self``, for method chaining.
        """
        self.squash = Squash(lim=lim, alpha=alpha, scale=scale, fn=fn)
        return self

    def disable_squash(self) -> "PrimitiveSet":
        """Remove the squash function so gene outputs are passed raw.

        Returns
        -------
        PrimitiveSet
            ``self``, for method chaining.
        """
        self.squash = None
        return self



    @classmethod
    def from_description(
        cls,
        description: str,
        client: Any,
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
        """Compile into a DEAP ``PrimitiveSet`` ready for evolution.

        Renames the default ``ARG0..N`` terminals to :attr:`feature_names`,
        registers all primitives and ephemerals, and returns the compiled
        DEAP object. Called internally by :class:`~symgene.Population`.

        Returns
        -------
        deap.gp.PrimitiveSet
            Configured DEAP primitive set.

        Raises
        ------
        ValueError
            If any ephemeral uses an unknown distribution.
        """
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
