import re

def n_nodes(individual) -> int:
    return sum(len(g) for g in individual)

def complexity_penalty(lambda_: float = 5e-4, tree_max_ref: int = 80):
    def pen(individual, X, y) -> float:
        n_genes = len(individual)
        total = n_nodes(individual)
        norm = total / max(1, n_genes * tree_max_ref)
        return lambda_ * norm
    return pen

def missing_vars_penalty(required: set[str], beta: float = 0.01):
    def pen(individual, X, y) -> float:
        used = set()
        for gene in individual:
            used.update(re.findall(r"\b(?:x\d+|\w+\d+)\b", str(gene)))
        used_req = used & required
        n_missing = len(required - used_req)
        return beta * (n_missing / max(1, len(required)))
    return pen
