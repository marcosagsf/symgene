"""Expression tree visualization for MGGP genes (pure matplotlib, no networkx)."""
import matplotlib.pyplot as plt


def _parse_tree(tree) -> dict:
    """Convert DEAP PrimitiveTree (prefix list) to children dict {node_idx: [child_idxs]}."""
    n = len(tree)
    children = {i: [] for i in range(n)}
    stack = []  # [(node_idx, remaining_children_needed)]

    for i, node in enumerate(tree):
        if stack:
            parent = stack[-1][0]
            children[parent].append(i)
            stack[-1][1] -= 1
            while stack and stack[-1][1] == 0:
                stack.pop()

        arity = node.arity if hasattr(node, "arity") else 0
        if arity > 0:
            stack.append([i, arity])

    return children


def _layout(children: dict, root: int = 0) -> dict:
    """Compute {node_idx: (x, y)} positions via subtree-width assignment."""
    width_cache: dict = {}

    def subtree_width(node: int) -> int:
        if node not in width_cache:
            cs = children.get(node, [])
            width_cache[node] = max(1, sum(subtree_width(c) for c in cs))
        return width_cache[node]

    pos: dict = {}

    def assign(node: int, x_start: float, depth: int) -> None:
        w = subtree_width(node)
        pos[node] = (x_start + w / 2.0, float(-depth))
        offset = x_start
        for child in children.get(node, []):
            cw = subtree_width(child)
            assign(child, offset, depth + 1)
            offset += cw

    assign(root, 0.0, 0)
    return pos


def plot_tree(tree, feature_names: list | None = None, ax=None, title: str = ""):
    """Plot a single DEAP PrimitiveTree as a matplotlib axes.

    Parameters
    ----------
    tree : DEAP PrimitiveTree or list of nodes with .name and .arity
    feature_names : list[str], optional
    ax : matplotlib Axes, optional — if None, creates a new figure and shows it
    title : str
    """
    children = _parse_tree(tree)
    pos = _layout(children)

    labels: dict = {}
    for i, node in enumerate(tree):
        name = node.name if hasattr(node, "name") else str(node)
        if feature_names and name.startswith("x") and name[1:].isdigit():
            idx = int(name[1:]) - 1
            if 0 <= idx < len(feature_names):
                name = feature_names[idx]
        labels[i] = name

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6, len(tree) * 0.5), 5))

    for node, (px, py) in pos.items():
        for child in children.get(node, []):
            cx, cy = pos[child]
            ax.plot([px, cx], [py, cy], "k-", lw=1, zorder=1)

    for node, (x, y) in pos.items():
        is_leaf = not children.get(node)
        color = "#AED6F1" if is_leaf else "#F9E79F"
        ax.text(
            x, y, labels[node],
            ha="center", va="center", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, edgecolor="grey"),
            zorder=2,
        )

    ax.set_axis_off()
    if title:
        ax.set_title(title)
    if fig is not None:
        plt.tight_layout()
        plt.show()
    return ax


def plot_individual_trees(individual, feature_names: list | None = None, max_genes: int = 4):
    """Plot up to max_genes gene trees from an SGIndividual side by side."""
    genes = list(individual)[:max_genes]
    n = len(genes)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    for i, (gene, ax) in enumerate(zip(genes, axes)):
        plot_tree(gene, feature_names=feature_names, ax=ax, title=f"Gene {i + 1}")
    plt.tight_layout()
    plt.show()
