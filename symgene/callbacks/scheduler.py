from symgene.callbacks.base import Callback


class ParameterScheduler(Callback):
    def __init__(self, target, param: str, schedule: dict):
        self.target = target
        self.param = param
        self.schedule = {int(k): v for k, v in schedule.items()}

    def on_generation_end(self, gen: int, logs: dict | None = None) -> None:
        for gen_key in sorted(self.schedule.keys()):
            if gen >= gen_key:
                setattr(self.target, self.param, self.schedule[gen_key])


class ReduceOnPlateau(Callback):
    def __init__(
        self,
        target,
        param: str,
        monitor: str,
        factor: float = 0.5,
        patience: int = 20,
        min_delta: float = 1e-6,
        min_value: float | None = None,
        mode: str = "auto",
    ):
        self.target = target
        self.param = param
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_delta = min_delta
        self.min_value = min_value
        self.mode = mode
        self._best = None
        self._wait = 0

    def on_train_begin(self, logs: dict | None = None) -> None:
        self._best = None
        self._wait = 0

    def on_generation_end(self, gen: int, logs: dict | None = None) -> None:
        if logs is None or self.monitor not in logs:
            return
        value = logs[self.monitor]
        minimize = not any(s in self.monitor.lower() for s in ("r2", "acc", "score")) \
            if self.mode == "auto" else self.mode == "min"

        if self._best is None:
            self._best = value
            return

        improved = (value < self._best - self.min_delta) if minimize \
            else (value > self._best + self.min_delta)

        if improved:
            self._best = value
            self._wait = 0
        else:
            self._wait += 1

        if self._wait >= self.patience:
            current = getattr(self.target, self.param)
            new_val = current * self.factor
            if self.min_value is not None:
                new_val = max(new_val, self.min_value)
            setattr(self.target, self.param, new_val)
            self._wait = 0
