"""SymGeneEvolver — multi-population evolutionary loop with migration support."""
from __future__ import annotations
import random
import numpy as np
import deap.tools as tools
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from symgene.results import SymGeneResult

from symgene.population import Population
from symgene.operators.crossover import intra_crossover
from symgene.operators.mutation import mutate_population
from symgene.metrics.regression import r2, mse


class SymGeneEvolver:
    """Multi-population evolutionary loop with optional migration and LLM rescue.

    Drives one or more :class:`~symgene.Population` instances through
    ``n_gen`` generations of selection, crossover, and mutation. Supports
    island-model evolution (multiple populations with periodic migration),
    early stopping via callbacks, and LLM-guided Genetic Rescue.

    Parameters
    ----------
    populations : list of Population
        One or more populations to evolve in parallel.
    n_gen : int
        Maximum number of generations. Default ``200``.
    cross_population : bool
        Enable inter-population gene exchange each generation. Default ``False``.
    cxpb_inter : float
        Fraction of individuals exchanged per inter-population crossover event.
        Default ``0.025``.
    seed : int, optional
        Global random seed (applied to both Python ``random`` and NumPy).
    n_jobs : int
        Reserved for future parallelisation (not yet implemented). Default ``1``.
    callbacks : list of Callback, optional
        Training callbacks (e.g. ``EarlyStopping``, ``Logger``).
    checkpoint_dir : str, optional
        Directory to write generation checkpoints (``.sgk`` files).
    checkpoint_every : int
        Checkpoint interval in generations. Default ``50``.
    verbose : int
        Verbosity level (``0`` = silent, ``1`` = milestones,
        ``2`` = every generation). Default ``1``.
    migration : bool
        Enable periodic island migration. Default ``False``.
    migration_freq : int
        Migration interval in generations. Default ``50``.
    migration_size : int
        Number of individuals transferred per migration event. Default ``1``.
    migration_topology : {"ring", "full", "best_to_worst"}
        Network topology for migration. Default ``"ring"``.
    migration_selection : str
        Which individuals to send (``"best"``). Default ``"best"``.
    migration_replace : str
        Which individuals to overwrite (``"worst"``). Default ``"worst"``.
    llm_client : LLMClient, optional
        Configured LLM client for Genetic Rescue (Phase 3).
    llm_context : dict, optional
        Per-population :class:`~symgene.llm.LLMContext` objects,
        keyed by population name.
    llm_rescue : bool
        Enable Genetic Rescue. Default ``False``.
    llm_rescue_fraction : float
        Fraction of worst individuals replaced per rescue event. Default ``0.1``.
    llm_rescue_trigger : {"stagnation", "random", "both"}
        Condition that triggers rescue. Default ``"stagnation"``.
    llm_rescue_level : {"gene", "individual"}
        Granularity of rescue substitution. Default ``"gene"``.
    llm_stagnation_patience : int
        Generations without improvement before stagnation rescue triggers.
        Default ``20``.
    llm_rescue_prob : float
        Probability of random rescue per generation (used with
        ``trigger="random"`` or ``"both"``). Default ``0.1``.
    llm_max_retries : int
        Maximum LLM retry attempts per rescue event. Default ``3``.

    Examples
    --------
    Single population (fastest setup):

    >>> import numpy as np
    >>> from symgene import PrimitiveSet, Population, SymGeneEvolver, FitnessEvaluator
    >>> from symgene.metrics.regression import mse
    >>> rng = np.random.default_rng(1)
    >>> X = rng.standard_normal((50, 2))
    >>> y = X[:, 0] + X[:, 1]
    >>> pset = PrimitiveSet(n_inputs=2).add_from_catalog(["add", "mul"])
    >>> pop = Population("demo", pset, n_genes=2, pop_size=10,
    ...                  fitness=FitnessEvaluator(metric=mse))
    >>> evolver = SymGeneEvolver([pop], n_gen=3, seed=0, verbose=0)
    >>> result = evolver.fit(X, {"demo": y})
    >>> "demo" in result
    True
    """

    def __init__(
        self,
        populations: list[Population],
        n_gen: int = 200,
        cross_population: bool = False,
        cxpb_inter: float = 0.025,
        seed: int | None = None,
        n_jobs: int = 1,
        callbacks: list[Any] | None = None,
        checkpoint_dir: str | None = None,
        checkpoint_every: int = 50,
        verbose: int = 1,
        migration: bool = False,
        migration_freq: int = 50,
        migration_size: int = 1,
        migration_topology: str = "ring",
        migration_selection: str = "best",
        migration_replace: str = "worst",
        # LLM Genetic Rescue
        llm_client: Any = None,
        llm_context: dict[str, Any] | None = None,
        llm_rescue: bool = False,
        llm_rescue_fraction: float = 0.1,
        llm_rescue_trigger: str = "stagnation",
        llm_rescue_level: str = "gene",
        llm_stagnation_patience: int = 20,
        llm_rescue_prob: float = 0.1,
        llm_max_retries: int = 3,
    ):
        self.populations = populations
        self.n_gen = n_gen
        self.cross_population = cross_population
        self.cxpb_inter = cxpb_inter
        self.seed = seed
        self.n_jobs = n_jobs
        self.callbacks = callbacks or []
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_every = checkpoint_every
        self.verbose = verbose
        self.migration = migration
        self.migration_freq = migration_freq
        self.migration_size = migration_size
        self.migration_topology = migration_topology
        self.migration_selection = migration_selection
        self.migration_replace = migration_replace
        self.llm_client = llm_client
        self.llm_context = llm_context or {}
        self.llm_rescue = llm_rescue
        self.llm_rescue_fraction = llm_rescue_fraction
        self.llm_rescue_trigger = llm_rescue_trigger
        self.llm_rescue_level = llm_rescue_level
        self.llm_stagnation_patience = llm_stagnation_patience
        self.llm_rescue_prob = llm_rescue_prob
        self.llm_max_retries = llm_max_retries

    def _build_logs(self, gen: int, pop_histories: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        logs = {"gen": gen}
        for pop_name, history in pop_histories.items():
            if history:
                entry = history[-1]
                for k, v in entry.items():
                    if k != "gen":
                        logs[f"{pop_name}_{k}"] = v
        return logs

    def fit(
        self,
        X: np.ndarray,
        y: dict[str, np.ndarray],
        X_val: np.ndarray | None = None,
        y_val: dict[str, np.ndarray] | None = None,
    ) -> SymGeneResult:
        """Run the evolutionary loop and return fitted results.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training input matrix, shared across all populations.
        y : dict of str to np.ndarray
            Mapping ``{population_name: target_array}`` for each population.
        X_val : np.ndarray, optional
            Validation inputs for tracking ``val_r2`` per generation.
        y_val : dict of str to np.ndarray, optional
            Validation targets. Required if ``X_val`` is given.

        Returns
        -------
        SymGeneResult
            Dict-like object mapping population name to
            :class:`~symgene.results.PopulationResult`.
        """
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        for pop in self.populations:
            pop.initialize(seed=self.seed)
            pop.evaluate(X, y[pop.name])

        pop_histories: dict[str, list[dict[str, Any]]] = {pop.name: [] for pop in self.populations}

        _llm_disabled = False
        _stagnation_counters = {pop.name: 0 for pop in self.populations}
        _best_fitness_seen = {pop.name: float("inf") for pop in self.populations}

        for cb in self.callbacks:
            cb.on_train_begin()

        try:
            for gen in range(self.n_gen):
                for pop in self.populations:
                    pop.apply_schedule(gen)
                    self._evolve_population(pop, X, y[pop.name])

                if self.cross_population and len(self.populations) > 1:
                    self._interpop_step(X, y)

                if (self.migration
                        and len(self.populations) > 1
                        and gen > 0
                        and gen % self.migration_freq == 0):
                    from symgene.operators.migration import migrate
                    migrate(
                        self.populations,
                        topology=self.migration_topology,
                        size=self.migration_size,
                        selection=self.migration_selection,
                        replace=self.migration_replace,
                    )
                    if self.verbose >= 2:
                        print(f"  Gen {gen}: migration ({self.migration_topology},"
                              f" size={self.migration_size})")

                if self.llm_rescue and not _llm_disabled and self.llm_client:
                    api_error = self._apply_rescue(gen, _stagnation_counters)
                    if api_error:
                        _llm_disabled = True

                for pop in self.populations:
                    pop.evaluate(X, y[pop.name])

                for pop in self.populations:
                    current_best = (
                        pop.best.fitness.values[0] if pop.best else float("inf")
                    )
                    if current_best < _best_fitness_seen[pop.name] - 1e-8:
                        _best_fitness_seen[pop.name] = current_best
                        _stagnation_counters[pop.name] = 0
                    else:
                        _stagnation_counters[pop.name] += 1

                for pop in self.populations:
                    best = pop.best
                    valid_fits = [
                        ind.fitness.values[0]
                        for ind in pop._population
                        if ind.fitness.valid and ind.fitness.values[0] < 1e6
                    ]
                    entry = {
                        "gen": gen,
                        "train_mse": best.fitness.values[0] if best else 1e9,
                        "mean_train_mse": float(np.mean(valid_fits)) if valid_fits else 1e9,
                        "std_train_mse": float(np.std(valid_fits)) if valid_fits else 0.0,
                        "n_genes": len(best) if best else 0,
                    }
                    if X_val is not None and y_val is not None:
                        y_pred_val = self._predict_individual(best, pop, X_val)
                        entry["val_r2"] = (
                            r2(y_val[pop.name], y_pred_val)
                            if y_pred_val is not None
                            else None
                        )
                    pop_histories[pop.name].append(entry)

                if self.verbose >= 1 and gen % max(1, self.n_gen // 20) == 0:
                    for pop in self.populations:
                        best_fit = pop.best.fitness.values[0] if pop.best else "?"
                        print(f"Gen {gen:4d} | {pop.name} | fitness={best_fit:.6f}")

                if self.checkpoint_dir and (gen + 1) % self.checkpoint_every == 0:
                    self._save_checkpoint(gen)

                logs = self._build_logs(gen, pop_histories)
                stop_signals = [cb.on_generation_end(gen, logs) for cb in self.callbacks]
                if any(s is True for s in stop_signals):
                    if self.verbose >= 1:
                        print(f"  Early stopping at generation {gen}")
                    break
        finally:
            for cb in self.callbacks:
                try:
                    cb.on_train_end()
                except Exception:
                    pass

        from symgene.results import SymGeneResult, PopulationResult
        return SymGeneResult({
            pop.name: PopulationResult(pop, pop_histories[pop.name], self)
            for pop in self.populations
        })

    def _evolve_population(self, pop: Population, X: np.ndarray, y: np.ndarray) -> None:
        """Apply one generation of selection, crossover, and mutation to ``pop``."""
        assert pop._toolbox is not None
        tb = pop._toolbox
        n_elite = pop.n_elite
        elite = list(map(tb.clone, tools.selBest(pop._population, n_elite)))

        n_offspring = len(pop._population) - n_elite
        offspring = list(map(
            tb.clone,
            pop.selection.select(
                pop._population,
                k=n_offspring,
                fitness_attr="fitness",
                minimize=True,
            )
        ))

        for i in range(0, len(offspring) - 1, 2):
            if random.random() < pop.cxpb:
                offspring[i], offspring[i + 1] = intra_crossover(
                    offspring[i], offspring[i + 1], tb,
                    cxpb_low=pop.cxpb_low, tree_max=pop.tree_max,
                )

        mutate_population(
            offspring, tb,
            mutpb=pop.mutpb, mutpb_low=pop.mutpb_low,
            mutation_weights=pop.mutation_weights,
            n_genes_max=pop.n_genes_max, tree_max=pop.tree_max,
        )

        pop._population[:] = offspring + elite

    def _interpop_step(self, X: np.ndarray, y: dict[str, np.ndarray]) -> None:
        """Perform inter-population crossover events between all population pairs."""
        from symgene.operators.crossover import interpop_crossover
        pairs = [
            (self.populations[i], self.populations[j])
            for i in range(len(self.populations))
            for j in range(i + 1, len(self.populations))
        ]
        for popA, popB in pairs:
            n_events = max(1, int(len(popA._population) * self.cxpb_inter))
            n_non_elite_A = len(popA._population) - popA.n_elite
            n_non_elite_B = len(popB._population) - popB.n_elite
            for _ in range(n_events):
                a = random.randint(0, n_non_elite_A - 1)
                b = random.randint(0, n_non_elite_B - 1)
                popA._population[a], popB._population[b], *_ = interpop_crossover(
                    popA._population[a], popB._population[b],
                    popA._toolbox, cxpb_low_inter=0.3, tree_max=popA.tree_max,
                )

    def _predict_individual(self, ind: Any, pop: Population, X: np.ndarray) -> np.ndarray | None:
        """Compile and evaluate ``ind`` on ``X`` using ``pop``'s DEAP pset.

        Returns ``None`` if the individual is invalid or compilation fails.
        """
        if ind is None or not hasattr(ind, '_combiner'):
            return None
        try:
            import deap.gp as gp
            funcs = [gp.compile(gene, pop._deap_pset) for gene in ind]
            G = np.column_stack([[f(*row) for row in X] for f in funcs])
            if not np.all(np.isfinite(G)):
                G = np.nan_to_num(G, nan=0.0, posinf=0.0, neginf=0.0)
            return ind._combiner.predict(G)
        except Exception:
            return None

    def _apply_rescue(
        self,
        gen: int,
        stagnation_counters: dict,
    ) -> bool:
        """Run Genetic Rescue for all populations that meet the trigger condition.

        Returns True if the LLM client raised an API error (disables for run).
        """
        from symgene.llm.rescue import rescue_worst

        for pop in self.populations:
            ctx = self.llm_context.get(pop.name)
            if ctx is None:
                continue

            trigger = self.llm_rescue_trigger
            should_rescue = False
            if trigger in ("stagnation", "both"):
                if stagnation_counters[pop.name] >= self.llm_stagnation_patience:
                    should_rescue = True
            if trigger in ("random", "both"):
                if random.random() < self.llm_rescue_prob:
                    should_rescue = True

            if not should_rescue:
                continue

            try:
                n = rescue_worst(
                    pop,
                    self.llm_client,
                    ctx,
                    rescue_fraction=self.llm_rescue_fraction,
                    level=self.llm_rescue_level,
                    max_retries=self.llm_max_retries,
                )
                if n > 0 and self.verbose >= 1:
                    print(
                        f"  Gen {gen}: Genetic Rescue — {n} individual(s) rescued"
                        f" in '{pop.name}'"
                    )
                if trigger in ("stagnation", "both"):
                    stagnation_counters[pop.name] = 0
            except Exception as exc:
                print(
                    f"\n[SymGene LLM] WARNING: API error in Genetic Rescue"
                    f" (gen {gen}, pop '{pop.name}'): {exc}\n"
                    f"  Genetic Rescue disabled for this run."
                    f" Evolution continues normally.\n"
                )
                return True  # signal: disable LLM

        return False  # LLM still active

    def _save_checkpoint(self, gen: int) -> None:
        """Serialize the full evolver state to ``checkpoint_dir/gen_NNNN.sgk``."""
        import os
        assert self.checkpoint_dir is not None
        try:
            import dill as _pickle
        except ImportError:
            import pickle as _pickle
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.checkpoint_dir, f"gen_{gen:04d}.sgk")
        with open(path, "wb") as f:
            _pickle.dump(self, f)
        if self.verbose >= 1:
            print(f"  Checkpoint saved: {path}")

    @classmethod
    def resume(cls, path: str) -> "SymGeneEvolver":
        """Load an evolver previously saved by :meth:`_save_checkpoint`.

        Parameters
        ----------
        path : str
            Path to a ``.sgk`` checkpoint file.

        Returns
        -------
        SymGeneEvolver
            Deserialized evolver instance.
        """
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)
