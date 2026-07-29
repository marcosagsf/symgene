"""Population-level statistics plots."""
import matplotlib.pyplot as plt
import numpy as np


def plot_n_genes_over_time(history: list, pop_name: str = "", ax=None):
    gens = [h["gen"] for h in history if "n_genes" in h]
    values = [h["n_genes"] for h in history if "n_genes" in h]
    if not gens:
        return
    fig = None
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(gens, values, marker="o", markersize=3)
    ax.set_xlabel("Generation")
    ax.set_ylabel("n_genes (best individual)")
    ax.set_title(f"Gene growth — {pop_name}" if pop_name else "Gene growth")
    if fig is not None:
        plt.tight_layout()
        plt.show()


def plot_gene_weights(coefficients, pop_name: str = "", ax=None):
    coefficients = np.asarray(coefficients)
    fig = None
    if ax is None:
        fig, ax = plt.subplots()
    x = np.arange(len(coefficients))
    ax.bar(x, np.abs(coefficients), color="#5DADE2", edgecolor="grey")
    ax.set_xlabel("Gene index")
    ax.set_ylabel("|coefficient|")
    ax.set_title(f"Gene weights — {pop_name}" if pop_name else "Gene weights")
    if fig is not None:
        plt.tight_layout()
        plt.show()


def plot_complexity_distribution(individuals: list, ax=None):
    complexities = [sum(len(g) for g in ind) for ind in individuals]
    fig = None
    if ax is None:
        fig, ax = plt.subplots()
    n_bins = min(20, max(1, len(set(complexities))))
    ax.hist(complexities, bins=n_bins, color="#A9DFBF", edgecolor="grey")
    ax.set_xlabel("Total nodes")
    ax.set_ylabel("Count")
    ax.set_title("Complexity distribution")
    if fig is not None:
        plt.tight_layout()
        plt.show()
