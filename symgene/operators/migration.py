import random
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


def migrate(
    populations: list,
    topology: str = "ring",
    size: int = 1,
    selection: str = "best",
    replace: str = "worst",
):
    """Island model migration: copy complete individuals between populations."""
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

    else:
        raise ValueError(f"Unknown migration topology: '{topology}'. "
                         f"Choose 'ring', 'full', or 'best_to_worst'.")
