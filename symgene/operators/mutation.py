import random
import deap.gp as gp
import deap.creator as creator


def mutate_population(
    population, toolbox, mutpb: float, mutpb_low: float,
    mutation_weights: list[float], n_genes_max: int, tree_max: int,
) -> tuple:
    low_mut = high_mut = add_ops = rem_ops = rep_ops = 0

    for i, ind in enumerate(population):
        if random.random() >= mutpb:
            continue

        individual = toolbox.clone(ind)

        if random.random() < mutpb_low:
            # low-order: subtree mutation on random genes
            n_mut = random.randint(1, max(1, int(len(individual) * 0.3)))
            for gene_idx in random.sample(range(len(individual)), min(n_mut, len(individual))):
                backup = creator.SGGene(individual[gene_idx])
                toolbox.mutate(individual[gene_idx])
                if len(individual[gene_idx]) > tree_max:
                    individual[gene_idx] = backup
            low_mut += 1
        else:
            # high-order: add / remove / replace genes
            action = random.choices(["remove", "add", "replace"], weights=mutation_weights, k=1)[0]
            if action == "remove" and len(individual) > 1:
                del individual[random.randint(0, len(individual) - 1)]
                rem_ops += 1
            elif action == "add" and len(individual) < n_genes_max:
                individual.append(toolbox.gene())
                add_ops += 1
            elif action == "replace" and len(individual) > 0:
                n_rep = random.randint(1, max(1, int(len(individual) * 0.3)))
                for idx in random.sample(range(len(individual)), n_rep):
                    individual[idx] = toolbox.gene()
                rep_ops += n_rep
            high_mut += 1

        if individual.fitness.valid:
            del individual.fitness.values
        population[i] = individual

    return population, low_mut, high_mut, add_ops, rem_ops, rep_ops
