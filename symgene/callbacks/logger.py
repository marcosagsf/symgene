from symgene.callbacks.base import Callback


class GenerationLogger(Callback):
    def __init__(
        self,
        metrics: list[str] | None = None,
        every: int = 1,
        file: str | None = None,
    ):
        self.metrics = metrics
        self.every = every
        self.file = file
        self._fh = None

    def on_train_begin(self, logs: dict | None = None) -> None:
        if self.file:
            self._fh = open(self.file, "w")

    def on_generation_end(self, gen: int, logs: dict | None = None) -> None:
        if logs is None or gen % self.every != 0:
            return
        parts = [f"gen={gen}"]
        for k, v in logs.items():
            if k == "gen":
                continue
            if self.metrics is None or k in self.metrics:
                parts.append(f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}")
        msg = " | ".join(parts)
        print(msg)
        if self._fh:
            self._fh.write(msg + "\n")
            self._fh.flush()

    def on_train_end(self, logs: dict | None = None) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None
