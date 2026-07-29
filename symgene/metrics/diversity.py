import numpy as np

def genotypic_diversity(population) -> float:
    exprs = [str(ind) for ind in population]
    return len(set(exprs)) / max(1, len(exprs))

def phenotypic_diversity(population, X: np.ndarray) -> float:
    if not hasattr(population[0], "predict"):
        return 0.0
    preds = np.array([ind.predict(X) for ind in population])
    return float(np.mean(np.var(preds, axis=0)))
