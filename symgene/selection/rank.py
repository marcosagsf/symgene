import numpy as np


class RankSelection:
    def __init__(self, pressure: float = 1.5):
        self.pressure = pressure  # 1.0 (uniform) to 2.0 (max bias)

    def select(
        self,
        population: list,
        k: int,
        fitness_attr: str = "fitness_value",
        minimize: bool = True,
    ) -> list:
        n = len(population)
        fitnesses = [getattr(ind, fitness_attr) for ind in population]
        ranked_indices = sorted(range(n), key=lambda i: fitnesses[i], reverse=not minimize)
        ranks = np.zeros(n)
        for rank_pos, orig_idx in enumerate(ranked_indices):
            ranks[orig_idx] = rank_pos + 1  # rank 1 = best
        probs = (2 - self.pressure) / n + \
                2 * (ranks - 1) * (self.pressure - 1) / (n * (n - 1))
        probs = np.clip(probs, 0.0, None)
        probs /= probs.sum()
        indices = np.random.choice(n, size=k, replace=True, p=probs)
        return [population[i] for i in indices]
