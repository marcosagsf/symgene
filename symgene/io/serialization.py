from symgene.results import SymGeneResult


def save_result(result: SymGeneResult, path: str) -> None:
    result.save(path)


def load_result(path: str) -> SymGeneResult:
    return SymGeneResult.load(path)
