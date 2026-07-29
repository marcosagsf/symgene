import numpy as np
from symgene.primitive_set import PrimitiveSet
from symgene.primitives.catalog import STANDARD
from symgene.population import Population
from symgene.evolver import SymGeneEvolver
from symgene.fitness import FitnessEvaluator
from symgene.metrics.regression import mse


class SymGeneRegressor:
    """High-level single-output MGGP regressor."""

    def __init__(
        self,
        n_genes: int = 8,
        pop_size: int = 100,
        n_gen: int = 200,
        primitives=None,
        squash: dict | None = None,
        combiner: str = "ridge",
        ridge_alphas: list | None = None,
        regression_degree: int = 1,
        feature_names: list[str] | None = None,
        seed: int | None = None,
        verbose: int = 1,
        **population_kwargs,
    ):
        self.n_genes = n_genes
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.primitives = primitives if primitives is not None else STANDARD
        self.squash = squash
        self.combiner = combiner
        self.ridge_alphas = ridge_alphas
        self.regression_degree = regression_degree
        self.feature_names = feature_names
        self.seed = seed
        self.verbose = verbose
        self._population_kwargs = population_kwargs
        self._result = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "SymGeneRegressor":
        n_inputs = X.shape[1]
        pset = PrimitiveSet(
            n_inputs=n_inputs,
            feature_names=self.feature_names,
        )
        pset.add_from_catalog(self.primitives)
        if self.squash:
            pset.set_squash(**self.squash)

        pop_kwargs = {k: v for k, v in self._population_kwargs.items()
                      if k not in ("ridge_alphas",)}
        pop = Population(
            name="_target",
            pset=pset,
            n_genes=self.n_genes,
            pop_size=self.pop_size,
            combiner=self.combiner,
            ridge_alphas=self.ridge_alphas,
            regression_degree=self.regression_degree,
            fitness=FitnessEvaluator(metric=mse),
            **pop_kwargs,
        )

        evolver = SymGeneEvolver(
            populations=[pop],
            n_gen=self.n_gen,
            seed=self.seed,
            verbose=self.verbose,
        )

        y_dict = {"_target": y}
        y_val_dict = {"_target": y_val} if y_val is not None else None

        full_results = evolver.fit(X, y_dict, X_val=X_val, y_val=y_val_dict)
        self._result = full_results["_target"]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._result.predict(X)

    def __getattr__(self, name: str):
        if name.startswith("_") or self._result is None:
            raise AttributeError(name)
        return getattr(self._result, name)
