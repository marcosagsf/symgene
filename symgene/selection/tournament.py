import random


def _scalar(val) -> float:
    """Extract a float from either a DEAP Fitness object or a plain scalar.
    Returns inf for invalid fitness (empty values tuple) so invalid individuals
    are never selected in a minimize-tournament.
    """
    if hasattr(val, 'values'):
        return float(val.values[0]) if val.values else float('inf')
    return float(val)


class TournamentSelection:
    def __init__(self, size: int = 5):
        self.size = size

    def select(self, population: list, k: int, fitness_attr: str = "fitness", minimize: bool = True) -> list:
        selected = []
        for _ in range(k):
            contestants = random.sample(population, min(self.size, len(population)))
            key_fn = lambda ind: _scalar(getattr(ind, fitness_attr))  # noqa: E731
            best = min(contestants, key=key_fn) if minimize \
                   else max(contestants, key=key_fn)
            selected.append(best)
        return selected
