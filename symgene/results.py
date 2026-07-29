import os
import numpy as np
import deap.gp as gp

try:
    import dill as pickle
except ImportError:
    import pickle


class PopulationResult:
    def __init__(self, population, history: list[dict], evolver):
        self._pop = population
        self._evolver = evolver
        self.history_ = history

    @property
    def best_individual_(self):
        return self._pop.best

    @property
    def best_expression_(self) -> str:
        ind = self.best_individual_
        if ind is None: return ""
        return " | ".join(str(gene) for gene in ind)

    @property
    def n_genes_(self) -> int:
        ind = self.best_individual_
        return len(ind) if ind is not None else 0

    @property
    def coefficients_(self) -> np.ndarray | None:
        ind = self.best_individual_
        if ind is None or not hasattr(ind, '_combiner'): return None
        return ind._combiner.coef_

    @property
    def best_fitness_(self) -> float:
        ind = self.best_individual_
        return ind.fitness.values[0] if ind is not None else float('inf')

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._evolver._predict_individual(self.best_individual_, self._pop, X)

    def to_string(self) -> str:
        return self.best_expression_

    def to_sympy(self):
        try:
            import sympy
            ind = self.best_individual_
            exprs = []
            for gene in ind:
                expr_str = str(gene)
                for i, name in enumerate(self._pop.pset.feature_names):
                    expr_str = expr_str.replace(name, f"var_{i}")
                sym_vars = {f"var_{i}": sympy.Symbol(name)
                            for i, name in enumerate(self._pop.pset.feature_names)}
                exprs.append(sympy.sympify(expr_str, locals=sym_vars))
            return exprs
        except Exception as e:
            return str(e)

    def to_latex(self) -> str:
        try:
            import sympy
            exprs = self.to_sympy()
            if isinstance(exprs, list):
                return " + ".join(sympy.latex(e) for e in exprs)
            return str(exprs)
        except Exception:
            return self.best_expression_

    def to_callable(self):
        ind = self.best_individual_
        pop = self._pop
        if ind is None: return None
        compiled = [gp.compile(gene, pop._deap_pset) for gene in ind]
        combiner = ind._combiner

        def fn(X: np.ndarray) -> np.ndarray:
            G = np.column_stack([[f(*row) for row in X] for f in compiled])
            return combiner.predict(G)
        return fn

    def plot_convergence(self, ax=None):
        import matplotlib.pyplot as plt
        gens = [h["gen"] for h in self.history_]
        fitness = [h["train_mse"] for h in self.history_]
        fig, ax_ = (None, ax) if ax else plt.subplots()
        ax_.plot(gens, fitness, label="train_mse")
        if self.history_ and "val_r2" in self.history_[0]:
            val_r2 = [h.get("val_r2") for h in self.history_]
            ax2 = ax_.twinx()
            ax2.plot(gens, val_r2, color="orange", label="val_r2")
            ax2.set_ylabel("val R²")
        ax_.set_xlabel("Generation")
        ax_.set_ylabel("Train MSE")
        ax_.set_title(f"Convergence — {self._pop.name}")
        if ax is None: plt.tight_layout(); plt.show()

    def plot_prediction(self, X: np.ndarray, y: np.ndarray, ax=None):
        import matplotlib.pyplot as plt
        y_pred = self.predict(X)
        fig, ax_ = (None, ax) if ax else plt.subplots()
        ax_.scatter(y, y_pred, alpha=0.5, s=10)
        lims = [min(y.min(), y_pred.min()), max(y.max(), y_pred.max())]
        ax_.plot(lims, lims, "r--", linewidth=1)
        ax_.set_xlabel("y true"); ax_.set_ylabel("y pred")
        ax_.set_title(f"Prediction — {self._pop.name}")
        if ax is None: plt.tight_layout(); plt.show()

    def plot_pareto(self, X: np.ndarray, y: np.ndarray, ax=None):
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

    def plot_expression_tree(self, gene_idx: int = 0, ax=None):
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

    @property
    def pareto_front_(self) -> list:
        hof = list(self._pop._hof)
        if not hof: return []
        complexities = [sum(len(g) for g in ind) for ind in hof]
        errors = [ind.fitness.values[0] for ind in hof]
        front_idx = self._pareto_front_indices(complexities, errors)
        return [hof[i] for i in front_idx]

    def best_by_accuracy(self):
        return self.best_individual_

    def best_by_simplicity(self, max_error: float = 0.1):
        front = self.pareto_front_
        candidates = [ind for ind in front if ind.fitness.values[0] <= max_error]
        if not candidates: return front[-1] if front else self.best_individual_
        return min(candidates, key=lambda ind: sum(len(g) for g in ind))

    def best_by_pareto(self, weight: float = 0.5):
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
    """Dict-like container: results["population_name"] -> PopulationResult."""

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "result.sgr"), "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "SymGeneResult":
        with open(os.path.join(path, "result.sgr"), "rb") as f:
            return pickle.load(f)
