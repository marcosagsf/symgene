"""SymGeneEvolver — multi-population evolutionary loop with migration support."""
import random
import numpy as np
import deap.tools as tools

from symgene.population import Population
from symgene.operators.crossover import intra_crossover
from symgene.operators.mutation import mutate_population
from symgene.metrics.regression import r2, mse


class SymGeneEvolver:
    def __init__(
        self,
        populations: list[Population],
        n_gen: int = 200,
        cross_population: bool = False,
        cxpb_inter: float = 0.025,
        seed: int | None = None,
        n_jobs: int = 1,
        callbacks: list | None = None,
        checkpoint_dir: str | None = None,
        checkpoint_every: int = 50,
        verbose: int = 1,
        migration: bool = False,
        migration_freq: int = 50,
        migration_size: int = 1,
        migration_topology: str = "ring",
        migration_selection: str = "best",
        migration_replace: str = "worst",
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

    def _build_logs(self, gen: int, pop_histories: dict) -> dict:
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
    ):
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        for pop in self.populations:
            pop.initialize(seed=self.seed)
            pop.evaluate(X, y[pop.name])

        pop_histories = {pop.name: [] for pop in self.populations}

        for cb in self.callbacks:
            cb.on_train_begin()

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

            for pop in self.populations:
                pop.evaluate(X, y[pop.name])

            for pop in self.populations:
                best = pop.best
                entry = {
                    "gen": gen,
                    "train_mse": best.fitness.values[0] if best else 1e9,
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

        for cb in self.callbacks:
            cb.on_train_end()

        from symgene.results import SymGeneResult, PopulationResult
        return SymGeneResult({
            pop.name: PopulationResult(pop, pop_histories[pop.name], self)
            for pop in self.populations
        })

    def _evolve_population(self, pop: Population, X, y):
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

    def _interpop_step(self, X, y):
        from symgene.operators.crossover import interpop_crossover
        pairs = [
            (self.populations[i], self.populations[j])
            for i in range(len(self.populations))
            for j in range(i + 1, len(self.populations))
        ]
        for popA, popB in pairs:
            n_events = max(1, int(len(popA._population) * self.cxpb_inter))
            n_non_elite = len(popA._population) - popA.n_elite
            for _ in range(n_events):
                a = random.randint(0, n_non_elite - 1)
                b = random.randint(0, n_non_elite - 1)
                popA._population[a], popB._population[b], *_ = interpop_crossover(
                    popA._population[a], popB._population[b],
                    popA._toolbox, cxpb_low_inter=0.3, tree_max=popA.tree_max,
                )

    def _predict_individual(self, ind, pop: Population, X: np.ndarray):
        if ind is None or not hasattr(ind, '_combiner'):
            return None
        try:
            import deap.gp as gp
            funcs = [gp.compile(gene, pop._deap_pset) for gene in ind]
            G = np.column_stack([[f(*row) for row in X] for f in funcs])
            return ind._combiner.predict(G)
        except Exception:
            return None

    def _save_checkpoint(self, gen: int):
        import os
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
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)
