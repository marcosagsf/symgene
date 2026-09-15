from __future__ import annotations
import os
import numpy as np
import deap.gp as gp
from typing import Any, Callable

try:
    import dill as pickle
except ImportError:
    import pickle


class PopulationResult:
    """Results and analysis tools for one evolved population.

    Produced by :meth:`~symgene.SymGeneEvolver.fit`; accessed via
    ``result["population_name"]``.

    Attributes
    ----------
    history_ : list of dict
        Per-generation log entries with keys ``gen``, ``train_mse``,
        ``mean_train_mse``, ``std_train_mse``, ``n_genes``, and optionally
        ``val_r2``.
    """

    def __init__(self, population: Any, history: list[dict[str, Any]], evolver: Any) -> None:
        self._pop = population
        self._evolver = evolver
        self.history_ = history

    @property
    def best_individual_(self):
        """Best individual found across all generations (Hall of Fame rank-0)."""
        return self._pop.best

    @property
    def best_expression_(self) -> str:
        """String representation of the best individual's genes, joined by \" | \"."""
        ind = self.best_individual_
        if ind is None: return ""
        return " | ".join(str(gene) for gene in ind)

    @property
    def n_genes_(self) -> int:
        """Number of gene trees in the best individual."""
        ind = self.best_individual_
        return len(ind) if ind is not None else 0

    @property
    def coefficients_(self) -> np.ndarray | None:
        """Linear combination weights for the best individual's genes.

        Returns ``None`` if no individual has been evaluated yet.
        """
        ind = self.best_individual_
        if ind is None or not hasattr(ind, '_combiner'): return None
        return ind._combiner.coef_

    @property
    def best_fitness_(self) -> float:
        """Fitness score of the best individual (lower is better)."""
        ind = self.best_individual_
        return ind.fitness.values[0] if ind is not None else float('inf')

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._evolver._predict_individual(self.best_individual_, self._pop, X)

    def to_string(self) -> str:
        return self.best_expression_

    def to_sympy(self) -> list[Any]:
        """Convert the best individual's gene trees to SymPy expressions.

        Returns
        -------
        list of sympy.Expr
            One SymPy expression per gene. Returns an empty list if the
            best individual is ``None`` or conversion fails.

        Warns
        -----
        UserWarning
            If any primitive lacks a ``sympy_fn`` mapping (registered via
            :meth:`PrimitiveSet.add_custom`).
        """
        try:
            import sympy as sp
            import deap.gp as gp
            import warnings

            ind = self.best_individual_
            if ind is None:
                return []

            registry = self._pop.pset.sympy_registry()
            feature_names = self._pop.pset.feature_names
            sym_vars = {name: sp.Symbol(name) for name in feature_names}

            def _eval(nodes, idx):
                node = nodes[idx]
                if isinstance(node, gp.Primitive):
                    sfn = registry.get(node.name)
                    args = []
                    next_idx = idx + 1
                    for _ in range(node.arity):
                        arg, next_idx = _eval(nodes, next_idx)
                        args.append(arg)
                    if sfn is None:
                        warnings.warn(
                            f"Primitiva '{node.name}' não possui sympy_fn. "
                            "Forneça sympy_fn em add_custom() para habilitar to_sympy().",
                            UserWarning,
                            stacklevel=4,
                        )
                        return None, next_idx
                    if any(a is None for a in args):
                        return None, next_idx
                    return sfn(*args), next_idx
                else:
                    name = node.name
                    if name in sym_vars:
                        return sym_vars[name], idx + 1
                    try:
                        return sp.Float(float(node.value)), idx + 1
                    except (TypeError, ValueError, AttributeError):
                        return sp.Symbol(str(name)), idx + 1

            exprs = []
            for gene in ind:
                expr, _ = _eval(list(gene), 0)
                exprs.append(expr)
            return exprs
        except Exception as exc:
            import warnings
            warnings.warn(
                f"to_sympy() failed ({type(exc).__name__}: {exc}); returning empty list.",
                UserWarning,
                stacklevel=2,
            )
            return []

    def to_latex(self) -> str:
        try:
            import sympy
            exprs = self.to_sympy()
            if isinstance(exprs, list) and exprs:
                parts = [sympy.latex(e) for e in exprs if e is not None]
                result = " + ".join(p for p in parts if p)
                if result:
                    return result
        except Exception as exc:
            import warnings
            warnings.warn(
                f"to_latex() failed ({type(exc).__name__}: {exc}); "
                "falling back to best_expression_.",
                UserWarning,
                stacklevel=2,
            )
        return self.best_expression_

    def to_callable(self) -> Callable[[np.ndarray], np.ndarray] | None:
        """Return a standalone Python callable wrapping the best individual.

        The returned function has the same behaviour as :meth:`predict` but
        does not hold a reference to the Population or Evolver — useful for
        exporting the model.

        Returns
        -------
        Callable[[np.ndarray], np.ndarray] or None
            Function ``f(X) -> y_pred``, or ``None`` if no individual exists.
        """
        ind = self.best_individual_
        pop = self._pop
        if ind is None: return None
        compiled = [gp.compile(gene, pop._deap_pset) for gene in ind]
        combiner = ind._combiner

        def fn(X: np.ndarray) -> np.ndarray:
            G = np.column_stack([[f(*row) for row in X] for f in compiled])
            return combiner.predict(G)
        return fn

    def plot_convergence(self, ax: Any = None) -> None:
        import matplotlib.pyplot as plt
        gens = [h["gen"] for h in self.history_]
        fitness = [h["train_mse"] for h in self.history_]
        fig, ax_ = (None, ax) if ax else plt.subplots()
        ax_.plot(gens, fitness, label="train_mse")
        if self.history_ and any("val_r2" in h for h in self.history_):
            val_r2: list[Any] = [h.get("val_r2") for h in self.history_]
            ax2 = ax_.twinx()
            ax2.plot(gens, val_r2, color="orange", label="val_r2")
            ax2.set_ylabel("val R²")
        ax_.set_xlabel("Generation")
        ax_.set_ylabel("Train MSE")
        ax_.set_title(f"Convergence — {self._pop.name}")
        if ax is None: plt.tight_layout(); plt.show()

    def plot_prediction(self, X: np.ndarray, y: np.ndarray, ax: Any = None) -> None:
        import matplotlib.pyplot as plt
        y_pred = self.predict(X)
        fig, ax_ = (None, ax) if ax else plt.subplots()
        ax_.scatter(y, y_pred, alpha=0.5, s=10)
        lims = [min(y.min(), y_pred.min()), max(y.max(), y_pred.max())]
        ax_.plot(lims, lims, "r--", linewidth=1)
        ax_.set_xlabel("y true"); ax_.set_ylabel("y pred")
        ax_.set_title(f"Prediction — {self._pop.name}")
        if ax is None: plt.tight_layout(); plt.show()

    def plot_pareto(self, X: np.ndarray, y: np.ndarray, ax: Any = None) -> None:
        import matplotlib.pyplot as plt
        from symgene.metrics.regression import mse as mse_fn
        hof = list(self._pop._hof)
        complexities, errors = [], []
        for ind in hof:
            c = sum(len(g) for g in ind)
            y_pred = self._evolver._predict_individual(ind, self._pop, X)
            e = mse_fn(y, y_pred) if y_pred is not None else 1e9
            complexities.append(c); errors.append(e)

        front_idx = self._pareto_front_indices(complexities, errors)
        fig, ax_ = (None, ax) if ax else plt.subplots()
        ax_.scatter(complexities, errors, alpha=0.5, s=20, label="HOF")
        fx = [complexities[i] for i in front_idx]
        fy = [errors[i] for i in front_idx]
        ax_.scatter(fx, fy, color="red", s=40, zorder=5, label="Pareto front")
        ax_.set_xlabel("Complexity (nodes)"); ax_.set_ylabel("MSE")
        ax_.set_title(f"Pareto Front — {self._pop.name}")
        ax_.legend()
        if ax is None: plt.tight_layout(); plt.show()

    def plot_expression_tree(self, gene_idx: int = 0, ax: Any = None) -> None:
        from symgene.visualization.expression import plot_tree
        ind = self.best_individual_
        if ind is None or len(ind) == 0:
            return
        gene_idx = min(gene_idx, len(ind) - 1)
        plot_tree(
            ind[gene_idx],
            feature_names=self._pop.pset.feature_names,
            ax=ax,
            title=f"Gene {gene_idx + 1} — {self._pop.name}",
        )

    def interpret(
        self,
        client: Any,
        description: str = "not provided",
        target_name: str | None = None,
    ) -> str:
        """Ask an LLM to physically interpret this population's best expression.

        Requires ``pip install symgene[llm]``.

        Parameters
        ----------
        client : LLMClient
            Configured LLM client from ``symgene.llm``.
        description : str
            Natural-language description of the modeled phenomenon.
            The richer the description, the more useful the interpretation.
        target_name : str or None
            Name of the target variable. Defaults to the population name.

        Returns
        -------
        str
            Physical interpretation generated by the LLM.

        Examples
        --------
        >>> from symgene.llm import LLMClient
        >>> client = LLMClient(provider="anthropic", model="claude-haiku-4-5-20251001")
        >>> text = result["PPF"].interpret(
        ...     client=client,
        ...     description="Peak Power Factor in a PWR reactor core",
        ...     target_name="PPF",
        ... )
        >>> print(text)
        """
        from symgene.llm.interpret import interpret_population
        return interpret_population(
            latex=self.to_latex(),
            description=description,
            target_name=target_name or self._pop.name,
            feature_names=self._pop.pset.feature_names,
            n_genes=self.n_genes_,
            coefficients=self.coefficients_,
            client=client,
        )

    def plot_gene_weights(self, ax: Any = None) -> None:
        from symgene.visualization.population_stats import plot_gene_weights
        coef = self.coefficients_
        if coef is not None:
            plot_gene_weights(coef, pop_name=self._pop.name, ax=ax)

    @property
    def pareto_front_(self) -> list:
        hof = list(self._pop._hof)
        if not hof: return []
        complexities = [sum(len(g) for g in ind) for ind in hof]
        errors = [ind.fitness.values[0] for ind in hof]
        front_idx = self._pareto_front_indices(complexities, errors)
        return [hof[i] for i in front_idx]

    def best_by_accuracy(self) -> Any:
        return self.best_individual_

    def best_by_simplicity(self, max_error: float = 0.1) -> Any:
        front = self.pareto_front_
        candidates = [ind for ind in front if ind.fitness.values[0] <= max_error]
        if not candidates: return front[-1] if front else self.best_individual_
        return min(candidates, key=lambda ind: sum(len(g) for g in ind))

    def best_by_pareto(self, weight: float = 0.5) -> Any:
        front = self.pareto_front_
        if not front: return self.best_individual_
        errors = np.array([ind.fitness.values[0] for ind in front])
        complexities = np.array([sum(len(g) for g in ind) for ind in front], float)
        e_norm = (errors - errors.min()) / (errors.max() - errors.min() + 1e-9)
        c_norm = (complexities - complexities.min()) / (complexities.max() - complexities.min() + 1e-9)
        scores = (1 - weight) * e_norm + weight * c_norm
        return front[int(np.argmin(scores))]

    @staticmethod
    def _pareto_front_indices(complexities, errors) -> list[int]:
        n = len(complexities)
        front = []
        for i in range(n):
            dominated = False
            for j in range(n):
                if i == j: continue
                if (complexities[j] <= complexities[i] and errors[j] <= errors[i] and
                        (complexities[j] < complexities[i] or errors[j] < errors[i])):
                    dominated = True; break
            if not dominated:
                front.append(i)
        return front


class SymGeneResult(dict):
    """Dict-like container mapping population names to their results.

    Returned by :meth:`~symgene.SymGeneEvolver.fit`. Inherits ``dict``,
    so population results are accessed as ``result["population_name"]``.

    Examples
    --------
    >>> result["PPF"].best_fitness_   # doctest: +SKIP
    0.00123
    >>> result["PPF"].to_latex()      # doctest: +SKIP
    'x_{1}^{2} + \\\\sin{x_{2}}'
    >>> result.save("/tmp/my_result") # doctest: +SKIP
    """

    def interpret(
        self,
        client: Any,
        descriptions: dict[str, str] | str = "not provided",
        target_names: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """Ask an LLM to interpret all populations in this result.

        Requires ``pip install symgene[llm]``.

        Parameters
        ----------
        client : LLMClient
            Configured LLM client from ``symgene.llm``.
        descriptions : dict or str
            Domain descriptions per population name, or a single string
            applied to all populations.
        target_names : dict or None
            Target variable names per population name. Defaults to
            population names.

        Returns
        -------
        dict of str to str
            ``{population_name: interpretation_text}``

        Examples
        --------
        >>> interpretations = result.interpret(
        ...     client=client,
        ...     descriptions={
        ...         "CBORON": "Critical boron concentration in a PWR",
        ...         "PPF": "Peak Power Factor in a PWR reactor core",
        ...     },
        ... )
        >>> for name, text in interpretations.items():
        ...     print(f"--- {name} ---")
        ...     print(text)
        """
        out: dict[str, str] = {}
        for name, pop_result in self.items():
            desc = (
                descriptions.get(name, "not provided")
                if isinstance(descriptions, dict)
                else descriptions
            )
            tname = (target_names or {}).get(name, name)
            out[name] = pop_result.interpret(
                client=client,
                description=desc,
                target_name=tname,
            )
        return out

    def save(self, path: str):
        """Serialize this result to ``path/result.sgr`` using dill.

        Parameters
        ----------
        path : str
            Directory path (created if it does not exist).
        """
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "result.sgr"), "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "SymGeneResult":
        """Deserialize a result saved by :meth:`save`.

        Parameters
        ----------
        path : str
            Directory containing ``result.sgr``.

        Returns
        -------
        SymGeneResult
            Loaded result object.
        """
        with open(os.path.join(path, "result.sgr"), "rb") as f:
            return pickle.load(f)
