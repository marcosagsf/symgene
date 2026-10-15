"""Tests for joblib-parallel fitness evaluation (n_jobs)."""
import time

import numpy as np
import pytest

from symgene.primitive_set import PrimitiveSet
from symgene.population import Population
from symgene.evolver import SymGeneEvolver
from symgene.fitness import FitnessEvaluator
from symgene.metrics.regression import mse


def _make_pop(n_jobs=1, backend="threading", pop_size=30, seed=0):
    pset = PrimitiveSet(n_inputs=2)
    pset.add_from_catalog(["add", "mul", "sub", "square", "sin"])
    pset.add_ephemeral("c", dist="uniform", low=-1.0, high=1.0, n=2)
    pop = Population(
        name="p", pset=pset, n_genes=4, pop_size=pop_size, tree_max=60,
        fitness=FitnessEvaluator(metric=mse),
        n_jobs=n_jobs, parallel_backend=backend,
    )
    pop.initialize(seed=seed)
    return pop


def _fitnesses(pop):
    return [ind.fitness.values[0] for ind in pop._population]


def test_population_parallel_matches_serial():
    """n_jobs=2 (threading) must produce exactly the same fitnesses as n_jobs=1.

    Both populations are initialised with the same seed, so they hold
    identical individuals; evaluation is deterministic, so the fitness
    vectors must match element for element.
    """
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, (80, 2))
    y = X[:, 0] ** 2 + np.sin(X[:, 1])

    serial = _make_pop(n_jobs=1, seed=123)
    serial.evaluate(X, y)

    parallel = _make_pop(n_jobs=2, backend="threading", seed=123)
    parallel.evaluate(X, y)

    np.testing.assert_allclose(_fitnesses(serial), _fitnesses(parallel), rtol=0, atol=0)


def test_population_parallel_updates_hof_and_attributes():
    """Parallel evaluation must still populate HOF and per-individual artifacts."""
    rng = np.random.default_rng(1)
    X = rng.uniform(-2, 2, (60, 2))
    y = X[:, 0] + X[:, 1]

    pop = _make_pop(n_jobs=2, backend="threading", seed=7)
    pop.evaluate(X, y)

    assert pop.best is not None
    for ind in pop._population:
        assert ind.fitness.valid
        # Successful evaluations attach a combiner and case errors.
        if ind.fitness.values[0] < 1e9:
            assert hasattr(ind, "_combiner")
            assert hasattr(ind, "_case_errors")


def test_evolver_propagates_n_jobs_to_populations():
    rng = np.random.default_rng(2)
    X = rng.uniform(-2, 2, (60, 2))
    y = X[:, 0] * X[:, 1]

    pset = PrimitiveSet(n_inputs=2)
    pset.add_from_catalog(["add", "mul", "sub"])
    pop = Population(name="demo", pset=pset, n_genes=3, pop_size=20,
                     fitness=FitnessEvaluator(metric=mse))
    evolver = SymGeneEvolver([pop], n_gen=3, seed=0, verbose=0, n_jobs=2)
    evolver.fit(X, {"demo": y})

    assert pop.n_jobs == 2


def test_evolver_parallel_matches_serial_best_fitness():
    rng = np.random.default_rng(3)
    X = rng.uniform(-2, 2, (70, 2))
    y = X[:, 0] ** 2 - X[:, 1]

    def run(n_jobs):
        pset = PrimitiveSet(n_inputs=2)
        pset.add_from_catalog(["add", "mul", "sub", "square"])
        pset.add_ephemeral("c", dist="uniform", low=-1.0, high=1.0, n=2)
        pop = Population(name="demo", pset=pset, n_genes=3, pop_size=25,
                         fitness=FitnessEvaluator(metric=mse))
        evolver = SymGeneEvolver([pop], n_gen=5, seed=42, verbose=0, n_jobs=n_jobs)
        result = evolver.fit(X, {"demo": y})
        return result["demo"].best_individual_.fitness.values[0]

    assert run(1) == pytest.approx(run(2), rel=1e-9)


def test_parallel_timing_smoke(capsys):
    """Serial and parallel both complete on a heavier workload (timing printed)."""
    rng = np.random.default_rng(4)
    X = rng.uniform(-2, 2, (200, 2))
    y = X[:, 0] ** 2 + np.sin(3 * X[:, 1])

    serial = _make_pop(n_jobs=1, pop_size=60, seed=99)
    t0 = time.perf_counter()
    serial.evaluate(X, y)
    t_serial = time.perf_counter() - t0

    parallel = _make_pop(n_jobs=2, backend="threading", pop_size=60, seed=99)
    t0 = time.perf_counter()
    parallel.evaluate(X, y)
    t_parallel = time.perf_counter() - t0

    with capsys.disabled():
        print(f"\n[n_jobs timing] serial={t_serial:.3f}s  "
              f"parallel(threading, 2)={t_parallel:.3f}s")

    # Correctness is the contract; speed-up is hardware-dependent and not asserted.
    np.testing.assert_allclose(_fitnesses(serial), _fitnesses(parallel))
