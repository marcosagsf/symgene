import random
import numpy as np


class LexicaseSelection:
    def __init__(self, epsilon=None):
        self.epsilon = epsilon

    def select(self, population, k, fitness_attr="fitness_value", minimize=True):
        n_cases = self._n_cases(population)
        selected = []
        for _ in range(k):
            pool = list(population)
            cases = list(range(n_cases))
            random.shuffle(cases)
            for case in cases:
                if len(pool) == 1:
                    break
                errors = [self._case_error(ind, case, fitness_attr) for ind in pool]
                errors = [float(e) for e in errors]
                threshold = min(errors) + (self.epsilon or 0.0)
                survivors = [ind for ind, e in zip(pool, errors) if e <= threshold]
                if survivors:
                    pool = survivors
                else:
                    break
            selected.append(random.choice(pool))
        return selected

    def _n_cases(self, population):
        for ind in population:
            errors = getattr(ind, "_case_errors", None)
            if errors is not None:
                return len(errors)
        return 1

    def _case_error(self, ind, case_idx, fitness_attr):
        errors = getattr(ind, "_case_errors", None)
        if errors is not None:
            return float(errors[case_idx])
        return float(getattr(ind, fitness_attr, 1e9))
