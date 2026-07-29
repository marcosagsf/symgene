from symgene.callbacks.base import Callback


class EarlyStopping(Callback):
    def __init__(
        self,
        monitor: str = "train_mse",
        patience: int = 50,
        min_delta: float = 1e-6,
        mode: str = "auto",
    ):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self._best = None
        self._wait = 0

    def on_train_begin(self, logs: dict | None = None) -> None:
        self._best = None
        self._wait = 0

    def on_generation_end(self, gen: int, logs: dict | None = None) -> bool | None:
        if logs is None or self.monitor not in logs:
            return None
        value = logs[self.monitor]
        minimize = self._resolve_minimize()

        if self._best is None:
            self._best = value
            self._wait = 0
            return None

        improved = (value < self._best - self.min_delta) if minimize \
            else (value > self._best + self.min_delta)

        if improved:
            self._best = value
            self._wait = 0
        else:
            self._wait += 1

        return True if self._wait >= self.patience else None

    def _resolve_minimize(self) -> bool:
        if self.mode == "min":
            return True
        if self.mode == "max":
            return False
        return not any(s in self.monitor.lower() for s in ("r2", "acc", "score"))
