import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest


def test_plot_n_genes_over_time_no_error():
    from symgene.visualization.population_stats import plot_n_genes_over_time
    history = [{"gen": i, "n_genes": 4 + i // 10} for i in range(30)]
    fig, ax = plt.subplots()
    plot_n_genes_over_time(history, pop_name="myp", ax=ax)
    plt.close("all")


def test_plot_n_genes_skips_missing_key():
    from symgene.visualization.population_stats import plot_n_genes_over_time
    history = [{"gen": i, "train_mse": 0.5} for i in range(10)]
    fig, ax = plt.subplots()
    plot_n_genes_over_time(history, ax=ax)
    plt.close("all")


def test_plot_gene_weights_no_error():
    from symgene.visualization.population_stats import plot_gene_weights
    coef = np.array([0.5, -0.3, 0.8, 0.1, -0.6])
    fig, ax = plt.subplots()
    plot_gene_weights(coef, pop_name="test", ax=ax)
    plt.close("all")


def test_plot_complexity_distribution_no_error():
    from symgene.visualization.population_stats import plot_complexity_distribution

    class FakeGene:
        def __len__(self): return 10

    class FakeInd:
        def __iter__(self): return iter([FakeGene(), FakeGene()])

    population = [FakeInd() for _ in range(20)]
    fig, ax = plt.subplots()
    plot_complexity_distribution(population, ax=ax)
    plt.close("all")


def test_evolver_history_contains_n_genes():
    import numpy as np
    from symgene import PrimitiveSet, Population, SymGeneEvolver
    from symgene.primitives import STANDARD
    pset = PrimitiveSet(n_inputs=2)
    pset.add_from_catalog(STANDARD)
    pop = Population("p", pset, n_genes=3, pop_size=8, n_genes_max=6)
    evolver = SymGeneEvolver(populations=[pop], n_gen=3, verbose=0)
    X = np.random.rand(20, 2)
    y = np.random.rand(20)
    results = evolver.fit(X, {"p": y})
    history = results["p"].history_
    assert "n_genes" in history[0]


def test_population_result_plot_gene_weights_no_error():
    import numpy as np
    from symgene import PrimitiveSet, Population, SymGeneEvolver
    from symgene.primitives import STANDARD
    pset = PrimitiveSet(n_inputs=2)
    pset.add_from_catalog(STANDARD)
    pop = Population("p", pset, n_genes=3, pop_size=8, n_genes_max=6)
    evolver = SymGeneEvolver(populations=[pop], n_gen=3, verbose=0)
    X = np.random.rand(20, 2)
    y = np.random.rand(20)
    results = evolver.fit(X, {"p": y})
    results["p"].plot_gene_weights()
    plt.close("all")
