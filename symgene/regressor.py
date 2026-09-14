import numpy as np
from typing import Any
from symgene.primitive_set import PrimitiveSet
from symgene.primitives.catalog import STANDARD
from symgene.population import Population
from symgene.evolver import SymGeneEvolver
from symgene.fitness import FitnessEvaluator
from symgene.metrics.regression import mse


class SymGeneRegressor:
    """High-level single-output MGGP regressor with sklearn-compatible API.

    Wraps a single :class:`~symgene.Population` and
    :class:`~symgene.SymGeneEvolver` into a fit/predict/score interface
    compatible with ``sklearn`` pipelines, ``cross_val_score``, and
    ``GridSearchCV``.

    Parameters
    ----------
    n_genes : int
        Number of gene trees per individual. Default ``8``.
    pop_size : int
        Population size. Default ``100``.
    n_gen : int
        Maximum number of generations. Default ``200``.
    primitives : list of str, optional
        Catalog keys for primitive functions. Defaults to ``STANDARD``.
    squash : dict, optional
        Keyword arguments forwarded to :meth:`PrimitiveSet.set_squash`.
        ``None`` disables squashing.
    combiner : {"ridge", "lasso", "linear"}
        Gene-combination strategy. Default ``"ridge"``.
    ridge_alphas : list of float, optional
        Regularization candidates for RidgeCombiner/LassoCombiner.
        Default ``[1.0, 5.0, 10.0]``.
    regression_degree : int
        Polynomial degree of the combiner feature matrix. Default ``1``.
    feature_names : list of str, optional
        Names for the input columns. Used in :meth:`to_sympy` /
        :meth:`to_latex` output.
    seed : int, optional
        Random seed for reproducibility.
    verbose : int
        Verbosity level (``0`` = silent, ``1`` = per-milestone,
        ``2`` = per-generation). Default ``1``.
    **population_kwargs
        Extra keyword arguments forwarded to :class:`~symgene.Population`.

    Examples
    --------
    >>> import numpy as np
    >>> from symgene import SymGeneRegressor
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((60, 2))
    >>> y = X[:, 0] ** 2 + X[:, 1]
    >>> reg = SymGeneRegressor(n_genes=2, pop_size=20, n_gen=5, seed=42, verbose=0)
    >>> _ = reg.fit(X, y)
    >>> reg.score(X, y) > 0
    True
    """

    def __init__(
        self,
        n_genes: int = 8,
        pop_size: int = 100,
        n_gen: int = 200,
        primitives: list[str] | None = None,
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
        self._result: Any = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "SymGeneRegressor":
        """Evolve the MGGP model on training data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training input matrix.
        y : np.ndarray of shape (n_samples,)
            Training target values.
        X_val : np.ndarray of shape (n_val, n_features), optional
            Validation inputs. If provided, ``val_r2`` is tracked per
            generation and available in the result history.
        y_val : np.ndarray of shape (n_val,), optional
            Validation targets. Required if ``X_val`` is given.

        Returns
        -------
        SymGeneRegressor
            ``self``, for method chaining (sklearn convention).
        """
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
        """Predict target values for new input data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input matrix.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted values from the best evolved individual.
        """
        return self._result.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute R² (coefficient of determination) on the given data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input matrix.
        y : np.ndarray of shape (n_samples,)
            True target values.

        Returns
        -------
        float
            R² score. Best possible is ``1.0``; can be negative for
            very poor models.
        """
        y_pred = self.predict(X)
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0

    def __sklearn_tags__(self):
        try:
            from sklearn.utils._tags import Tags, RegressorTags, TargetTags, InputTags
            return Tags(
                estimator_type="regressor",
                target_tags=TargetTags(required=True, multi_output=False),
                regressor_tags=RegressorTags(),
                input_tags=InputTags(allow_nan=False),
            )
        except (ImportError, TypeError):
            return {"estimator_type": "regressor"}

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return estimator parameters (sklearn ``BaseEstimator`` protocol).

        Parameters
        ----------
        deep : bool
            Ignored (no sub-estimators). Kept for sklearn compatibility.

        Returns
        -------
        dict
            All constructor parameters by name.
        """
        return {
            "n_genes": self.n_genes,
            "pop_size": self.pop_size,
            "n_gen": self.n_gen,
            "primitives": self.primitives,
            "squash": self.squash,
            "combiner": self.combiner,
            "ridge_alphas": self.ridge_alphas,
            "regression_degree": self.regression_degree,
            "feature_names": self.feature_names,
            "seed": self.seed,
            "verbose": self.verbose,
            **self._population_kwargs,
        }

    def set_params(self, **params: Any) -> "SymGeneRegressor":
        """Set estimator parameters (sklearn ``BaseEstimator`` protocol).

        Parameters
        ----------
        **params
            Parameter names and their new values. Unknown keys are stored
            in ``_population_kwargs`` and forwarded to
            :class:`~symgene.Population`.

        Returns
        -------
        SymGeneRegressor
            ``self``, for method chaining.
        """
        _named = {
            "n_genes", "pop_size", "n_gen", "primitives", "squash",
            "combiner", "ridge_alphas", "regression_degree",
            "feature_names", "seed", "verbose",
        }
        for key, value in params.items():
            if key in _named:
                setattr(self, key, value)
            else:
                self._population_kwargs[key] = value
        return self

    def __getattr__(self, name: str):
        if name.startswith("_") or self._result is None:
            raise AttributeError(name)
        return getattr(self._result, name)
