import numpy as np
from typing import Any, Callable


class FitnessEvaluator:
    """Computes individual fitness as a metric plus optional penalty terms.

    Parameters
    ----------
    metric : Callable[[np.ndarray, np.ndarray], float]
        Base fitness function ``metric(y_true, y_pred) -> float``.
        Lower is better (minimization). Use functions from
        ``symgene.metrics.regression`` (e.g. ``mse``, ``rmse``).
    penalties : list of Callable, optional
        Additional penalty functions with signature
        ``penalty(individual, X, y) -> float``. Each penalty is added to
        the base metric score. Defaults to no penalties.

    Examples
    --------
    >>> import numpy as np
    >>> from symgene.metrics.regression import mse
    >>> fe = FitnessEvaluator(metric=mse)
    >>> y = np.array([1.0, 2.0, 3.0])
    >>> y_pred = np.array([1.1, 1.9, 3.1])
    >>> fe.compute(None, None, y, y_pred) < 0.02
    True
    """

    def __init__(
        self,
        metric: Callable[[np.ndarray, np.ndarray], float],
        penalties: list[Callable[[Any, np.ndarray, np.ndarray], float]] | None = None,
    ) -> None:
        self.metric = metric
        self.penalties = penalties if penalties is not None else []

    def compute(
        self,
        individual: Any,
        X: np.ndarray | None,
        y: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        """Evaluate fitness for one individual.

        Parameters
        ----------
        individual : Any
            DEAP individual (passed to penalty functions). May be ``None``
            when called outside an evolutionary context.
        X : np.ndarray or None
            Input matrix (passed to penalty functions).
        y : np.ndarray
            True target values.
        y_pred : np.ndarray
            Predicted values produced by the individual.

        Returns
        -------
        float
            Scalar fitness score (lower is better). Equal to
            ``metric(y, y_pred) + sum(pen(individual, X, y) for pen in penalties)``.
        """
        score = self.metric(y, y_pred)
        for pen_fn in self.penalties:
            score += pen_fn(individual, X, y)
        return float(score)
