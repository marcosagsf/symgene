"""Standalone export functions — delegate to PopulationResult methods."""


def to_sympy(population_result):
    return population_result.to_sympy()


def to_latex(population_result) -> str:
    return population_result.to_latex()


def to_callable(population_result):
    return population_result.to_callable()


def to_string(population_result) -> str:
    return population_result.to_string()
