import random
import numpy as np
import deap.tools as tools
import deap.creator as creator


def _clone_individual(ind):
    new_ind = creator.SGIndividual(
        creator.SGGene(gene) for gene in ind
    )
    if ind.fitness.valid:
        new_ind.fitness.values = ind.fitness.values
    return new_ind


def _select_migrants(population, size: int, strategy: str) -> list:
    if strategy == "best":
        valid = [ind for ind in population if ind.fitness.valid]
        if not valid:
            valid = population
        return [_clone_individual(ind)
                for ind in random.sample(valid, min(size, len(valid)))]
    elif strategy == "random":
        return [_clone_individual(ind)
                for ind in random.sample(population, min(size, len(population)))]
    else:
        raise ValueError(f"Unknown migration selection strategy: '{strategy}'. "
                         f"Choose 'best' or 'random'.")


def _replace_in_population(population, migrants: list, strategy: str):
    k = min(len(migrants), len(population))
    if strategy == "worst":
        valid = [ind for ind in population if ind.fitness.valid]
        if not valid:
            # replace random indices when no fitness available
            indices = random.sample(range(len(population)), k)
            for idx, new in zip(indices, migrants):
                population[idx] = new
            return
        targets = tools.selWorst(valid, min(k, len(valid)))
        for old, new in zip(targets, migrants):
            idx = population.index(old)
            population[idx] = new
    elif strategy == "random":
        indices = random.sample(range(len(population)), k)
        for idx, new in zip(indices, migrants):
            population[idx] = new
    else:
        raise ValueError(f"Unknown migration replace strategy: '{strategy}'. "
                         f"Choose 'worst' or 'random'.")


def _migrate_along_adjacency(pops, adjacency, size, selection, replace):
    """Send migrants from i to j for every adjacency[i][j] that is True.

    Migrants are selected once per source population and the same batch is
    offered to each of its receivers, so a single good individual can spread
    to several islands in one round.
    """
    n = len(pops)
    for i in range(n):
        receivers = [j for j in range(n) if j != i and adjacency[i][j]]
        if not receivers:
            continue
        migrants = _select_migrants(pops[i], size, selection)
        for j in receivers:
            _replace_in_population(pops[j], migrants, replace)


def migrate(
    populations: list,
    topology: str = "ring",
    size: int = 1,
    selection: str = "best",
    replace: str = "worst",
    hub_index: int = 0,
    adjacency=None,
):
    """Island model migration: copy complete individuals between populations.

    Parameters
    ----------
    populations : list of Population
        The islands taking part in migration. Fewer than two is a no-op.
    topology : {"ring", "full", "best_to_worst", "star", "custom"}
        Migration pattern. ``"ring"`` passes migrants to the next island,
        ``"full"`` broadcasts to every other island, ``"best_to_worst"``
        seeds the weakest island from the strongest, ``"star"`` exchanges
        through a single hub, and ``"custom"`` follows an explicit
        adjacency matrix.
    size : int
        Number of individuals migrated per sending island.
    selection : {"best", "random"}
        How migrants are chosen from a sending island.
    replace : {"worst", "random"}
        Which individuals are overwritten in a receiving island.
    hub_index : int
        Index of the hub island for ``topology="star"``. Every other island
        (a spoke) sends migrants to the hub, and the hub sends migrants to
        every spoke. Ignored by other topologies.
    adjacency : np.ndarray, optional
        Required for ``topology="custom"``. A boolean array of shape
        ``(N, N)`` where ``N == len(populations)``. ``adjacency[i][j] = True``
        means island ``i`` sends migrants to island ``j``. The diagonal is
        ignored. This subsumes the other topologies as special cases.

    Examples
    --------
    >>> # ring built as a custom adjacency matrix (3 islands)
    >>> import numpy as np
    >>> adj = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=bool)
    >>> migrate(pops, topology="custom", adjacency=adj)  # doctest: +SKIP
    """
    n = len(populations)
    if n < 2:
        return

    pops = [p._population for p in populations]

    if topology == "ring":
        migrants_per_pop = [_select_migrants(p, size, selection) for p in pops]
        for i in range(n):
            receiver = pops[(i + 1) % n]
            _replace_in_population(receiver, migrants_per_pop[i], replace)

    elif topology == "full":
        for i in range(n):
            migrants = _select_migrants(pops[i], size, selection)
            for j in range(n):
                if i != j:
                    _replace_in_population(pops[j], migrants, replace)

    elif topology == "best_to_worst":
        def mean_fitness(pop):
            valid = [ind.fitness.values[0] for ind in pop if ind.fitness.valid]
            return sum(valid) / len(valid) if valid else float("inf")

        fitnesses = [mean_fitness(p) for p in pops]
        best_idx = fitnesses.index(min(fitnesses))
        worst_idx = fitnesses.index(max(fitnesses))
        if best_idx != worst_idx:
            migrants = _select_migrants(pops[best_idx], size, selection)
            _replace_in_population(pops[worst_idx], migrants, replace)

    elif topology == "star":
        if not 0 <= hub_index < n:
            raise ValueError(
                f"hub_index must be in [0, {n - 1}] for {n} populations, "
                f"got {hub_index}."
            )
        adjacency = np.zeros((n, n), dtype=bool)
        for spoke in range(n):
            if spoke == hub_index:
                continue
            adjacency[spoke][hub_index] = True   # spoke -> hub
            adjacency[hub_index][spoke] = True   # hub -> spoke
        _migrate_along_adjacency(pops, adjacency, size, selection, replace)

    elif topology == "custom":
        if adjacency is None:
            raise ValueError(
                "topology='custom' requires an 'adjacency' matrix."
            )
        adjacency = np.asarray(adjacency, dtype=bool)
        if adjacency.shape != (n, n):
            raise ValueError(
                f"adjacency must have shape ({n}, {n}) to match the number "
                f"of populations, got {adjacency.shape}."
            )
        _migrate_along_adjacency(pops, adjacency, size, selection, replace)

    else:
        raise ValueError(f"Unknown migration topology: '{topology}'. "
                         f"Choose 'ring', 'full', 'best_to_worst', 'star', "
                         f"or 'custom'.")
