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


def test_evolver_stops_on_early_stopping_signal():
    """When a callback returns True from on_generation_end, evolution stops early."""
    class StopAtGen2(Callback):
        def on_generation_end(self, gen, logs=None):
            return True if gen >= 2 else None

    import numpy as np
    from symgene import PrimitiveSet, Population, SymGeneEvolver
    from symgene.primitives import STANDARD
    pset = PrimitiveSet(n_inputs=2)
    pset.add_from_catalog(STANDARD)
    pop = Population("p", pset, n_genes=2, pop_size=10, n_genes_max=4)
    cb = StopAtGen2()
    evolver = SymGeneEvolver(populations=[pop], n_gen=20, callbacks=[cb], verbose=0)
    X = np.random.rand(20, 2)
    y = np.random.rand(20)
    results = evolver.fit(X, {"p": y})
    # If early stopping worked, history should have at most 3 entries (gen 0,1,2)
    history = results["p"].history_
    assert len(history) <= 3


from symgene.callbacks.early_stopping import EarlyStopping
from symgene.callbacks.scheduler import ParameterScheduler, ReduceOnPlateau


def test_early_stopping_triggers():
    cb = EarlyStopping(monitor="pop1_train_mse", patience=3, min_delta=1e-4)
    cb.on_train_begin()
    log_values = [0.5, 0.499, 0.498, 0.497, 0.497, 0.497, 0.497]
    results = [cb.on_generation_end(i, {"gen": i, "pop1_train_mse": v}) for i, v in enumerate(log_values)]
    assert True in results


def test_early_stopping_no_trigger_with_improvement():
    cb = EarlyStopping(monitor="pop1_train_mse", patience=5, min_delta=1e-4)
    cb.on_train_begin()
    result = None
    for i in range(10):
        result = cb.on_generation_end(i, {"gen": i, "pop1_train_mse": 1.0 / (i + 1)})
    assert result is None


def test_early_stopping_maximize_mode():
    cb = EarlyStopping(monitor="pop1_val_r2", patience=2, min_delta=1e-4, mode="max")
    cb.on_train_begin()
    results = [cb.on_generation_end(i, {"gen": i, "pop1_val_r2": 0.8}) for i in range(5)]
    assert True in results


def test_early_stopping_missing_monitor_ignored():
    cb = EarlyStopping(monitor="does_not_exist", patience=2)
    cb.on_train_begin()
    result = None
    for i in range(5):
        result = cb.on_generation_end(i, {"gen": i, "pop1_train_mse": 0.5})
    assert result is None


def test_early_stopping_resets_on_train_begin():
    cb = EarlyStopping(monitor="pop1_train_mse", patience=2)
    cb.on_train_begin()
    for i in range(5):
        cb.on_generation_end(i, {"gen": i, "pop1_train_mse": 0.5})
    cb.on_train_begin()
    result = cb.on_generation_end(0, {"gen": 0, "pop1_train_mse": 0.9})
    assert result is None


def test_parameter_scheduler_fires_at_gen():
    class FakePop:
        cxpb = 0.9
    pop = FakePop()
    cb = ParameterScheduler(target=pop, param="cxpb", schedule={0: 0.9, 5: 0.5, 10: 0.3})
    cb.on_generation_end(0, {})
    assert pop.cxpb == 0.9
    cb.on_generation_end(4, {})
    assert pop.cxpb == 0.9
    cb.on_generation_end(5, {})
    assert pop.cxpb == 0.5
    cb.on_generation_end(10, {})
    assert pop.cxpb == 0.3


def test_reduce_on_plateau_fires():
    class FakePop:
        mutpb = 0.4
    pop = FakePop()
    cb = ReduceOnPlateau(
        target=pop, param="mutpb", monitor="pop1_train_mse",
        factor=0.5, patience=3, min_delta=1e-6,
    )
    cb.on_train_begin()
    for i in range(10):
        cb.on_generation_end(i, {"gen": i, "pop1_train_mse": 0.5})
    assert pop.mutpb < 0.4


def test_reduce_on_plateau_min_value():
    class FakePop:
        lr = 0.1
    pop = FakePop()
    cb = ReduceOnPlateau(
        target=pop, param="lr", monitor="pop1_train_mse",
        factor=0.1, patience=1, min_value=0.01,
    )
    cb.on_train_begin()
    for i in range(20):
        cb.on_generation_end(i, {"gen": i, "pop1_train_mse": 0.5})
    assert pop.lr >= 0.01
