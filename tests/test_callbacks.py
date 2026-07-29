import pytest
from symgene.callbacks import Callback, GenerationLogger


def test_callback_base_noop():
    cb = Callback()
    cb.on_train_begin()
    assert cb.on_generation_end(0, {}) is None
    cb.on_train_end()


def test_generation_logger_every(capsys):
    cb = GenerationLogger(every=5)
    cb.on_train_begin()
    for gen in range(10):
        logs = {"gen": gen, "pop1_train_mse": 0.5 - gen * 0.05}
        cb.on_generation_end(gen, logs)
    captured = capsys.readouterr()
    lines = [l for l in captured.out.splitlines() if l]
    assert len(lines) == 2  # gen 0 and gen 5


def test_generation_logger_metrics_filter(capsys):
    cb = GenerationLogger(metrics=["pop1_train_mse"])
    cb.on_train_begin()
    logs = {"gen": 0, "pop1_train_mse": 0.5, "pop1_val_r2": 0.9}
    cb.on_generation_end(0, logs)
    captured = capsys.readouterr()
    assert "train_mse" in captured.out
    assert "val_r2" not in captured.out


def test_generation_logger_file(tmp_path):
    log_path = str(tmp_path / "run.log")
    cb = GenerationLogger(every=1, file=log_path)
    cb.on_train_begin()
    cb.on_generation_end(0, {"gen": 0, "pop1_train_mse": 0.5})
    cb.on_train_end()
    with open(log_path) as f:
        content = f.read()
    assert "gen=0" in content
    assert "train_mse" in content


def test_evolver_calls_on_train_begin_and_end():
    class TrackingCallback(Callback):
        def __init__(self):
            self.begin_count = 0
            self.end_count = 0
        def on_train_begin(self, logs=None):
            self.begin_count += 1
        def on_train_end(self, logs=None):
            self.end_count += 1

    import numpy as np
    from symgene import PrimitiveSet, Population, SymGeneEvolver
    from symgene.primitives import STANDARD
    pset = PrimitiveSet(n_inputs=2)
    pset.add_from_catalog(STANDARD)
    pop = Population("p", pset, n_genes=2, pop_size=10, n_genes_max=4)
    cb = TrackingCallback()
    evolver = SymGeneEvolver(populations=[pop], n_gen=2, callbacks=[cb], verbose=0)
    X = np.random.rand(20, 2)
    y = np.random.rand(20)
    evolver.fit(X, {"p": y})
    assert cb.begin_count == 1
    assert cb.end_count == 1


def test_evolver_logs_contain_population_metrics():
    class LogCapture(Callback):
        def __init__(self):
            self.last_logs = None
        def on_generation_end(self, gen, logs=None):
            self.last_logs = logs

    import numpy as np
    from symgene import PrimitiveSet, Population, SymGeneEvolver
    from symgene.primitives import STANDARD
    pset = PrimitiveSet(n_inputs=2)
    pset.add_from_catalog(STANDARD)
    pop = Population("myp", pset, n_genes=2, pop_size=10, n_genes_max=4)
    cb = LogCapture()
    evolver = SymGeneEvolver(populations=[pop], n_gen=2, callbacks=[cb], verbose=0)
    X = np.random.rand(20, 2)
    y = np.random.rand(20)
    evolver.fit(X, {"myp": y})
    assert cb.last_logs is not None
    assert "gen" in cb.last_logs
    assert "myp_train_mse" in cb.last_logs
