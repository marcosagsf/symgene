class Callback:
    def on_train_begin(self, logs: dict | None = None) -> None:
        pass

    def on_generation_end(self, gen: int, logs: dict | None = None) -> bool | None:
        return None

    def on_train_end(self, logs: dict | None = None) -> None:
        pass
