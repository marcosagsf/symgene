import numpy as np
import random

def select_genes_by_weight(weights: np.ndarray, k: int) -> list[int]:
    n = len(weights)
    k = min(k, n)
    valid = np.where(weights > 0)[0]
    if len(valid) == 0:
        return random.sample(range(n), k)
    k = min(k, len(valid))
    w = weights[valid].astype(float)
    total = w.sum()
    probs = w / total if total > 0 else np.ones(len(valid)) / len(valid)
    chosen = np.random.choice(valid, size=k, replace=False, p=probs)
    return chosen.tolist()
