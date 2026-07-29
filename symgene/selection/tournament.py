import random

class TournamentSelection:
    def __init__(self, size: int = 5):
        self.size = size

    def select(self, population: list, k: int, fitness_attr: str = "fitness_value", minimize: bool = True) -> list:
        selected = []
        for _ in range(k):
            contestants = random.sample(population, min(self.size, len(population)))
            best = min(contestants, key=lambda ind: getattr(ind, fitness_attr)) if minimize \
                   else max(contestants, key=lambda ind: getattr(ind, fitness_attr))
            selected.append(best)
        return selected
