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


class LowPerformanceTrigger(BaseTrigger):
    """Fires when val_r2 < target_r2 during the first `max_gen` generations.

    Unlike StagnationTrigger (which waits for a plateau), this trigger fires
    during the *active improvement phase* when the model hasn't reached a
    target quality yet. Useful for guiding the search early — before the GP
    settles into a local attractor — on problems that converge quickly to a
    decent-but-suboptimal solution.

    Parameters
    ----------
    target_r2 : float
        Fire while validation R² is below this value (default: 0.95).
    max_gen : int
        Stop firing after this generation — avoids interfering with late-stage
        fine-tuning (default: 80).
    min_gen : int
        Minimum generation before the trigger is eligible (default: 3).
    """

    def __init__(
        self,
        target_r2: float = 0.95,
        max_gen: int = 80,
        min_gen: int = 3,
    ) -> None:
        self.target_r2 = target_r2
        self.max_gen = max_gen
        self.min_gen = min_gen

    def evaluate(self, snapshot: MonitorSnapshot) -> bool:
        if snapshot.gen < self.min_gen or snapshot.gen > self.max_gen:
            return False
        if snapshot.val_r2 is None:
            return False
        return float(snapshot.val_r2) < self.target_r2
