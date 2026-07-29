"""Stub implementation of SymGeneResult / PopulationResult.

T16 will replace this with the full implementation including serialization,
ensemble methods, and extended reporting.
"""
import numpy as np


class PopulationResult:
    def __init__(self, population, history, evolver):
        self._pop = population
        self._evolver = evolver
        self.history_ = history

    @property
    def best_individual_(self):
        return self._pop.best

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._evolver._predict_individual(self.best_individual_, self._pop, X)


class SymGeneResult(dict):
    """Dict-like container mapping population name → PopulationResult."""
    pass
