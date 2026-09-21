import numpy as np
import pytest
from symgene.primitive_set import PrimitiveSet
from symgene.population import Population
from symgene.evolver import SymGeneEvolver
from symgene.fitness import FitnessEvaluator
from symgene.metrics.regression import mse

rng = np.random.default_rng(0)
X = rng.standard_normal((80, 3))
y = X[:, 0] ** 2 - X[:, 1] + 0.5

def make_pset():
    pset = PrimitiveSet(n_inputs=3)
    pset.add_from_catalog(["add", "sub", "mul", "square", "sin"])
    pset.set_squash(lim=8)
    return pset

def make_pop(name="p1"):
    return Population(
        name=name, pset=make_pset(), n_genes=3, pop_size=20,
        fitness=FitnessEvaluator(metric=mse), combiner="ridge",
    )

def test_evolver_single_pop_fit_returns_result():
    evolver = SymGeneEvolver(populations=[make_pop()], n_gen=3, seed=0)
    results = evolver.fit(X, {"p1": y})
    assert "p1" in results

def test_evolver_result_has_best():
    evolver = SymGeneEvolver(populations=[make_pop()], n_gen=3, seed=0)
    results = evolver.fit(X, {"p1": y})
    assert results["p1"].best_individual_ is not None

def test_evolver_result_predict():
    evolver = SymGeneEvolver(populations=[make_pop()], n_gen=3, seed=0)
    results = evolver.fit(X, {"p1": y})
    y_pred = results["p1"].predict(X)
    assert y_pred.shape == (80,)

def test_evolver_history_grows():
    evolver = SymGeneEvolver(populations=[make_pop()], n_gen=5, seed=0)
    results = evolver.fit(X, {"p1": y})
    assert len(results["p1"].history_) == 5

def test_evolver_with_val():
    X_val = rng.standard_normal((20, 3))
    y_val = X_val[:, 0] ** 2 - X_val[:, 1] + 0.5
    evolver = SymGeneEvolver(populations=[make_pop()], n_gen=3, seed=0)
    results = evolver.fit(X, {"p1": y}, X_val=X_val, y_val={"p1": y_val})
    hist = results["p1"].history_
    assert "val_r2" in hist[0]

def test_evolver_migration_preserves_pop_sizes():
    pop1 = make_pop("p1")
    pop2 = make_pop("p2")
    evolver = SymGeneEvolver(
        populations=[pop1, pop2], n_gen=5, seed=0,
        migration=True, migration_freq=2, migration_size=1,
        migration_topology="ring", migration_selection="best",
        migration_replace="worst", verbose=0,
    )
    y2 = rng.standard_normal(80)
    results = evolver.fit(X, {"p1": y, "p2": y2})
    assert len(pop1._population) == 20
    assert len(pop2._population) == 20

def test_evolver_single_pop_cross_population_flag():
    """cross_population=True with a single population must not crash."""
    evolver = SymGeneEvolver(
        populations=[make_pop()], n_gen=3, seed=0,
        cross_population=True, verbose=0,
    )
    results = evolver.fit(X, {"p1": y})
    assert "p1" in results

def test_evolver_single_pop_migration_flag():
    """migration=True with a single population must not crash."""
    evolver = SymGeneEvolver(
        populations=[make_pop()], n_gen=4, seed=0,
        migration=True, migration_freq=2, migration_size=1,
        migration_topology="ring", verbose=0,
    )
    results = evolver.fit(X, {"p1": y})
    assert "p1" in results

def test_evolver_two_pops_cross_population():
    """cross_population=True with two populations exercises _interpop_step."""
    pop1 = make_pop("p1")
    pop2 = make_pop("p2")
    y2 = rng.standard_normal(80)
    evolver = SymGeneEvolver(
        populations=[pop1, pop2], n_gen=5, seed=0,
        cross_population=True, cxpb_inter=0.5, verbose=0,
    )
    results = evolver.fit(X, {"p1": y, "p2": y2})
    assert "p1" in results and "p2" in results

def test_evolver_checkpoint_saves_file(tmp_path):
    """checkpoint_dir causes a .sgk file to be written and verbose=1 prints path."""
    evolver = SymGeneEvolver(
        populations=[make_pop()], n_gen=3, seed=0,
        checkpoint_dir=str(tmp_path), checkpoint_every=2, verbose=1,
    )
    evolver.fit(X, {"p1": y})
    sgk_files = list(tmp_path.glob("*.sgk"))
    assert len(sgk_files) >= 1

def test_evolver_resume_from_checkpoint(tmp_path):
    """resume() loads a checkpoint and the evolver is usable."""
    evolver = SymGeneEvolver(
        populations=[make_pop()], n_gen=3, seed=0,
        checkpoint_dir=str(tmp_path), checkpoint_every=2, verbose=0,
    )
    evolver.fit(X, {"p1": y})
    sgk_files = sorted(tmp_path.glob("*.sgk"))
    assert sgk_files, "no checkpoint was saved"
    loaded = SymGeneEvolver.resume(str(sgk_files[0]))
    assert hasattr(loaded, "populations")

def test_evolver_n_jobs_2_warns():
    """n_jobs != 1 emits a UserWarning."""
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        SymGeneEvolver(populations=[make_pop()], n_gen=1, seed=0, n_jobs=2)
    user_warns = [x for x in w if issubclass(x.category, UserWarning)]
    assert len(user_warns) > 0

def test_evolver_verbose2_migration():
    """verbose=2 with migration prints extra info without crashing."""
    pop1 = make_pop("p1")
    pop2 = make_pop("p2")
    y2 = rng.standard_normal(80)
    evolver = SymGeneEvolver(
        populations=[pop1, pop2], n_gen=4, seed=0,
        migration=True, migration_freq=2, migration_size=1,
        verbose=2,
    )
    evolver.fit(X, {"p1": y, "p2": y2})

def test_evolver_early_stopping_verbose1(capsys):
    """EarlyStopping with verbose=1 prints the early-stop message."""
    import numpy as np
    from symgene import PrimitiveSet, Population, SymGeneEvolver
    from symgene.primitives import STANDARD
    from symgene.callbacks.early_stopping import EarlyStopping
    pset = PrimitiveSet(n_inputs=2)
    pset.add_from_catalog(STANDARD)
    pop = Population("p1", pset, n_genes=2, pop_size=10, n_genes_max=4)
    cb = EarlyStopping(monitor="p1_train_mse", patience=1, min_delta=1e9, mode="min")
    X2 = np.random.default_rng(7).standard_normal((30, 2))
    y2 = X2[:, 0] ** 2
    evolver = SymGeneEvolver(populations=[pop], n_gen=20, callbacks=[cb], verbose=1)
    evolver.fit(X2, {"p1": y2})
    captured = capsys.readouterr()
    assert "Early stopping" in captured.out

def test_evolver_predict_individual_none():
    """_predict_individual returns None when ind is None."""
    evolver = SymGeneEvolver(populations=[make_pop()], n_gen=1, seed=0)
    result = evolver._predict_individual(None, make_pop(), X)
    assert result is None

def test_evolver_migration_disabled_by_default():
    pop1 = make_pop("p1")
    pop2 = make_pop("p2")
    evolver = SymGeneEvolver(
        populations=[pop1, pop2], n_gen=3, seed=0, verbose=0
    )
    y2 = rng.standard_normal(80)
    results = evolver.fit(X, {"p1": y, "p2": y2})
    assert "p1" in results and "p2" in results
