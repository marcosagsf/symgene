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
            ranks[orig_idx] = n - 1 - rank_pos  # rank=n-1 = best, rank=0 = worst
        # Linear rank selection: P(i) = (2-SP)/N + 2*i*(SP-1)/(N*(N-1)), i=rank
        probs = (2 - self.pressure) / n + \
                2 * ranks * (self.pressure - 1) / (n * (n - 1))
        probs = np.clip(probs, 0.0, None)
        probs /= probs.sum()
        indices = np.random.choice(n, size=k, replace=True, p=probs)
        return [population[i] for i in indices]
