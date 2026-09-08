from __future__ import annotations
from symgene.advisor.monitor import MonitorSnapshot


class BaseTrigger:
    def evaluate(self, snapshot: MonitorSnapshot) -> bool:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.__class__.__name__


class StagnationTrigger(BaseTrigger):
    def __init__(self, patience: int = 20) -> None:
        self.patience = patience

    def evaluate(self, snapshot: MonitorSnapshot) -> bool:
        return snapshot.stagnation_counter >= self.patience


class DiversityTrigger(BaseTrigger):
    def __init__(self, threshold: float = 0.15) -> None:
        self.threshold = threshold

    def evaluate(self, snapshot: MonitorSnapshot) -> bool:
        return snapshot.diversity_index < self.threshold


class BloatTrigger(BaseTrigger):
    def __init__(self, complexity_growth: float = 0.10) -> None:
        self.complexity_growth = complexity_growth

    def evaluate(self, snapshot: MonitorSnapshot) -> bool:
        return snapshot.complexity_trend > self.complexity_growth
