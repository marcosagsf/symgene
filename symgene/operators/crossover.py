import random
import deap.gp as gp
import deap.creator as creator
from symgene.selection.gene_selection import select_genes_by_weight
import numpy as np


def intra_crossover(indA, indB, toolbox, cxpb_low: float, tree_max: int):
    """Low-order (subtree) or high-order (whole gene swap) intra-population crossover."""
    if len(indA) == 0 or len(indB) == 0:
        return indA, indB

    k = random.randint(1, min(len(indA), len(indB)))
    idxA = random.sample(range(len(indA)), k)
    idxB = random.sample(range(len(indB)), k)

    if random.random() < cxpb_low:
        # low-order: subtree crossover
        for a, b in zip(idxA, idxB):
            if len(indA[a]) + len(indB[b]) - 1 <= tree_max * 2:
                indA[a], indB[b] = toolbox.mate(indA[a], indB[b])
    else:
        # high-order: swap whole genes
        for a, b in zip(idxA, idxB):
            indA[a], indB[b] = indB[b], indA[a]

    del indA.fitness.values
    del indB.fitness.values
    return indA, indB


def interpop_crossover(indA, indB, toolbox, cxpb_low_inter: float, tree_max: int):
    """Coefficient-weighted inter-population crossover."""
    if len(indA) == 0 or len(indB) == 0:
        return indA, indB, 0, 0, 0

    def get_weights(ind):
        if hasattr(ind, '_combiner') and hasattr(ind._combiner, 'gene_weights_'):
            return ind._combiner.gene_weights_
        return np.ones(len(ind))

    k = random.randint(1, min(len(indA), len(indB)))
    idxA = select_genes_by_weight(get_weights(indA), k)
    idxB = select_genes_by_weight(get_weights(indB), k)

    n_low = n_high = n_fallback = 0

    if random.random() < cxpb_low_inter:
        for a, b in zip(idxA, idxB):
            if len(indA[a]) + len(indB[b]) - 1 > tree_max:
                indA[a], indB[b] = indB[b], indA[a]
                n_high += 1; n_fallback += 1
            else:
                indA[a], indB[b] = toolbox.mate(indA[a], indB[b])
                n_low += 1
    else:
        for a, b in zip(idxA, idxB):
            indA[a], indB[b] = indB[b], indA[a]
            n_high += 1

    if indA.fitness.valid: del indA.fitness.values
    if indB.fitness.valid: del indB.fitness.values
    return indA, indB, n_low, n_high, n_fallback
