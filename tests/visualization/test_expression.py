import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest


def _make_node(name, arity):
    class Node:
        pass
    n = Node()
    n.name = name
    n.arity = arity
    return n


def _make_tree():
    """Represents: add(x1, mul(x2, x3)) in prefix order."""
    return [
        _make_node("add", 2),
        _make_node("x1", 0),
        _make_node("mul", 2),
        _make_node("x2", 0),
        _make_node("x3", 0),
    ]


def test_parse_tree_children():
    from symgene.visualization.expression import _parse_tree
    tree = _make_tree()
    children = _parse_tree(tree)
    assert children[0] == [1, 2]
    assert children[2] == [3, 4]
    assert children[1] == []


def test_layout_depth():
    from symgene.visualization.expression import _parse_tree, _layout
    tree = _make_tree()
    children = _parse_tree(tree)
    pos = _layout(children)
    assert len(pos) == 5
    assert pos[0][1] == 0.0
    assert pos[1][1] == -1.0
    assert pos[2][1] == -1.0
    assert pos[3][1] == -2.0


def test_layout_single_node():
    from symgene.visualization.expression import _parse_tree, _layout
    tree = [_make_node("x1", 0)]
    children = _parse_tree(tree)
    pos = _layout(children)
    assert len(pos) == 1
    assert pos[0] == (0.5, 0.0)


def test_plot_tree_no_error():
    from symgene.visualization.expression import plot_tree
    tree = _make_tree()
    fig, ax = plt.subplots()
    plot_tree(tree, ax=ax)
    plt.close("all")


def test_plot_tree_with_feature_names():
    from symgene.visualization.expression import plot_tree
    tree = _make_tree()
    fig, ax = plt.subplots()
    plot_tree(tree, feature_names=["alpha", "beta", "gamma"], ax=ax)
    plt.close("all")


def test_population_result_has_plot_expression_tree():
    import numpy as np
    from symgene import PrimitiveSet, Population, SymGeneEvolver
    from symgene.primitives import STANDARD
    pset = PrimitiveSet(n_inputs=2)
    pset.add_from_catalog(STANDARD)
    pop = Population("p", pset, n_genes=2, pop_size=8, n_genes_max=4)
    evolver = SymGeneEvolver(populations=[pop], n_gen=2, verbose=0)
    X = np.random.rand(20, 2)
    y = np.random.rand(20)
    results = evolver.fit(X, {"p": y})
    results["p"].plot_expression_tree()
    plt.close("all")
