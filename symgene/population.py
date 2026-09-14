import random
import numpy as np
import deap.gp as gp
import deap.base as base
import deap.creator as creator
import deap.tools as tools
from typing import Any

from symgene.primitive_set import PrimitiveSet
from symgene.fitness import FitnessEvaluator
from symgene.metrics.regression import mse
from symgene.selection.tournament import TournamentSelection
from symgene.combiners.ridge import RidgeCombiner
from symgene.combiners.lasso import LassoCombiner
from symgene.combiners.linear import LinearCombiner


def _get_combiner(name: str, **kwargs):
    if name == "ridge":
        return RidgeCombiner(**kwargs)
    if name == "lasso":
        return LassoCombiner(**kwargs)
    if name == "linear":
        return LinearCombiner()
    raise ValueError(f"Unknown combiner: {name}")


if not hasattr(creator, "SGFitnessMin"):
    creator.create("SGFitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "SGGene"):
    creator.create("SGGene", gp.PrimitiveTree, fitness=creator.SGFitnessMin)
if not hasattr(creator, "SGIndividual"):
    creator.create("SGIndividual", list, fitness=creator.SGFitnessMin)


class Population:
    """Single evolutionary population of multi-gene GP individuals.

    Manages the DEAP toolbox, individual initialization, fitness evaluation,
    and scheduling of hyperparameter changes over generations. Multiple
    ``Population`` instances can be combined in a
    :class:`~symgene.SymGeneEvolver` for island-model evolution.

    Parameters
    ----------
    name : str
        Unique identifier for this population. Used as the key in
        :class:`~symgene.SymGeneResult`.
    pset : PrimitiveSet
        Configured primitive set defining the search space.
    n_genes : int
        Initial number of gene trees per individual. Default ``8``.
    n_genes_max : int
        Maximum genes allowed after gene-addition mutations. Default ``30``.
    pop_size : int
        Number of individuals in the population. Default ``100``.
    tree_min : int
        Minimum total node count per individual (soft constraint). Default ``50``.
    tree_max : int
        Maximum total node count per individual (hard constraint). Default ``80``.
    height_max : int
        Maximum tree height used during mutation. Default ``8``.
    tree_init_max : int, optional
        Maximum tree height used during initialization. Defaults to
        ``height_max``.
    combiner : {"ridge", "lasso", "linear"}
        Strategy for combining gene outputs into a single prediction.
        Default ``"ridge"``.
    ridge_alphas : list of float, optional
        Regularization candidates for ``RidgeCombiner``/``LassoCombiner``.
        Default ``[1.0, 5.0, 10.0]``.
    regression_degree : int
        Polynomial degree applied to the gene-output matrix before fitting
        the combiner. Default ``1`` (linear in gene outputs).
    fitness : FitnessEvaluator, optional
        Fitness function wrapper. Defaults to ``FitnessEvaluator(metric=mse)``.
    selection : Any, optional
        Selection operator. Defaults to
        ``TournamentSelection(size=max(2, int(0.05 * pop_size)))``.
    elite_ratio : float
        Fraction of population preserved as elites each generation.
        Default ``0.025``.
    cxpb : float
        Intra-individual crossover probability per pair. Default ``0.975``.
    cxpb_low : float
        Per-gene crossover probability within intra-individual crossover.
        Default ``0.5``.
    mutpb : float
        Overall mutation probability per individual. Default ``0.2``.
    mutpb_low : float
        Per-gene mutation probability within overall mutation. Default ``0.2``.
    mutation_weights : list of float, optional
        Relative weights for the three mutation operators
        ``[point, subtree, gene_add]``. Default ``[1.0, 1.0, 1.2]``.
    schedule : dict, optional
        Generation-based hyperparameter schedule. Example::

            schedule={"cxpb": {100: 0.5, 150: 0.3}}

        sets ``cxpb=0.5`` from generation 100 and ``cxpb=0.3`` from 150.
    """

    def __init__(
        self,
        name: str,
        pset: PrimitiveSet,
        n_genes: int = 8,
        n_genes_max: int = 30,
        pop_size: int = 100,
        tree_min: int = 50,
        tree_max: int = 80,
        height_max: int = 8,
        tree_init_max: int | None = None,
        combiner: str = "ridge",
        ridge_alphas: list[float] | None = None,
        regression_degree: int = 1,
        fitness: FitnessEvaluator | None = None,
        selection: Any = None,
        elite_ratio: float = 0.025,
        cxpb: float = 0.975,
        cxpb_low: float = 0.5,
        mutpb: float = 0.2,
        mutpb_low: float = 0.2,
        mutation_weights: list[float] | None = None,
        schedule: dict | None = None,
    ):
        self.name = name
        self.pset = pset
        self.n_genes = n_genes
        self.n_genes_max = n_genes_max
        self.pop_size = pop_size
        self.tree_min = tree_min
        self.tree_max = tree_max
        self.height_max = height_max
        self.tree_init_max = tree_init_max if tree_init_max is not None else height_max
        self.combiner_name = combiner
        self.ridge_alphas = ridge_alphas or [1.0, 5.0, 10.0]
        self.regression_degree = regression_degree
        self.fitness = fitness or FitnessEvaluator(metric=mse)
        self.selection = selection or TournamentSelection(size=max(2, int(0.05 * pop_size)))
        self.elite_ratio = elite_ratio
        self.cxpb = cxpb
        self.cxpb_low = cxpb_low
        self.mutpb = mutpb
        self.mutpb_low = mutpb_low
        self.mutation_weights = mutation_weights or [1.0, 1.0, 1.2]
        self.schedule = schedule or {}

        self._deap_pset: gp.PrimitiveSet | None = None
        self._toolbox: base.Toolbox | None = None
        self._population: list[Any] = []
        self._hof: tools.HallOfFame | None = None
        self.history: list[dict[str, Any]] = []

    def initialize(self, seed: int | None = None) -> None:
        """Build the DEAP toolbox and create the initial population.

        Must be called before :meth:`evaluate` or any evolution step.

        Parameters
        ----------
        seed : int, optional
            Random seed applied before population generation.
        """
        if seed is not None:
            random.seed(seed)
        self._deap_pset = self.pset.build()
        self._setup_toolbox()
        assert self._toolbox is not None
        self._population = [self._toolbox.individual() for _ in range(self.pop_size)]
        self._hof = tools.HallOfFame(maxsize=10)

    def _setup_toolbox(self) -> None:
        self._toolbox = base.Toolbox()

        def gen_tree():
            return creator.SGGene(
                gp.genHalfAndHalf(self._deap_pset, min_=2, max_=self.tree_init_max)
            )

        def gen_tree_mut():
            # Genes added/replaced during mutation use height_max, not tree_init_max,
            # so evolution can explore deeper structures than the initial population.
            return creator.SGGene(
                gp.genHalfAndHalf(self._deap_pset, min_=2, max_=self.height_max)
            )

        def gen_individual():
            return creator.SGIndividual(gen_tree() for _ in range(self.n_genes))

        self._toolbox.register("gene", gen_tree)
        self._toolbox.register("gene_mut", gen_tree_mut)
        self._toolbox.register("individual", gen_individual)
        self._toolbox.register("mate", gp.cxOnePoint)
        mut_height = max(3, self.height_max // 2)
        self._toolbox.register(
            "expr_mut", gp.genHalfAndHalf,
            pset=self._deap_pset, min_=1, max_=mut_height
        )
        self._toolbox.register(
            "mutate", gp.mutUniform,
            expr=self._toolbox.expr_mut,
            pset=self._deap_pset,
        )
        self._toolbox.register("clone", lambda ind: creator.SGIndividual(
            creator.SGGene(g) for g in ind
        ))

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> None:
        """Evaluate fitness for all individuals and update the Hall of Fame.

        Individuals that raise exceptions during evaluation receive a
        fitness of ``1e9`` (worst possible) and are silently discarded.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input matrix.
        y : np.ndarray of shape (n_samples,)
            Target values.
        """
        combiner_kwargs: dict[str, Any] = {"degree": self.regression_degree}
        if self.combiner_name in ("ridge", "lasso"):
            combiner_kwargs["alphas"] = self.ridge_alphas

        for ind in self._population:
            self._eval_individual(ind, X, y, combiner_kwargs)

        assert self._hof is not None
        self._hof.update(self._population)

    def _eval_individual(self, ind: Any, X: np.ndarray, y: np.ndarray, combiner_kwargs: dict[str, Any]) -> None:
        try:
            funcs = [gp.compile(expr=gene, pset=self._deap_pset) for gene in ind]
            G = np.column_stack([
                [f(*row) for row in X] for f in funcs
            ])
            if not np.all(np.isfinite(G)):
                G = np.nan_to_num(G, nan=0.0, posinf=0.0, neginf=0.0)

            combiner = _get_combiner(self.combiner_name, **combiner_kwargs)
            combiner.fit(G, y)
            y_pred = combiner.predict(G)
            ind._case_errors = np.abs(y - y_pred)

            fitness_val = self.fitness.compute(ind, X, y, y_pred)
            ind.fitness.values = (fitness_val,)
            ind._combiner = combiner
            ind._gene_outputs_std = G.std(axis=0)
        except Exception:
            ind.fitness.values = (1e9,)

    def apply_schedule(self, generation: int) -> None:
        """Apply generation-based hyperparameter updates from :attr:`schedule`.

        Parameters
        ----------
        generation : int
            Current generation index (0-based).
        """
        for param, timeline in self.schedule.items():
            for gen_key in sorted(timeline.keys()):
                if generation >= gen_key:
                    setattr(self, param, timeline[gen_key])

    @property
    def best(self) -> Any:
        """Best individual found so far (Hall of Fame rank-0), or ``None``."""
        return self._hof[0] if self._hof else None

    @property
    def n_elite(self) -> int:
        """Number of elite individuals preserved each generation."""
        return max(1, int(self.elite_ratio * self.pop_size))
