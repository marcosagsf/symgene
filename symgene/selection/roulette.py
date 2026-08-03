import numpy as np


def _scalar(val) -> float:
    """Extract a float from either a DEAP Fitness object or a plain scalar."""
    return float(val.values[0]) if hasattr(val, 'values') else float(val)


class RouletteSelection:
    def select(
        self,
        population: list,
        k: int,
        fitness_attr: str = "fitness_value",
        minimize: bool = True,
    ) -> list:
        fitnesses = np.array([_scalar(getattr(ind, fitness_attr)) for ind in population], dtype=float)
        if minimize:
            fitnesses = fitnesses.max() - fitnesses + 1e-9
        fitnesses = np.clip(fitnesses, 0.0, None)
        total = fitnesses.sum()
        probs = fitnesses / total if total > 0 else np.ones(len(population)) / len(population)
        indices = np.random.choice(len(population), size=k, replace=True, p=probs)
        return [population[i] for i in indices]
