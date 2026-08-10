from unittest.mock import MagicMock

import numpy as np

from symgene.llm.interpret import interpret_population


def _mock_client(response: str = "Physical interpretation text.") -> MagicMock:
    client = MagicMock()
    client.complete.return_value = response
    return client


def test_interpret_population_returns_string():
    client = _mock_client("The expression represents a cosine flux profile.")
    result = interpret_population(
        latex=r"\cos(x_1) + x_2^2",
        description="axial neutron flux in a reactor",
        target_name="flux",
        feature_names=["axial_position", "burnup"],
        n_genes=2,
        coefficients=np.array([0.8, 0.2]),
        client=client,
    )
    assert isinstance(result, str)
    assert len(result) > 0


def test_interpret_population_passes_description_to_prompt():
    client = _mock_client()
    interpret_population(
        latex=r"x_1 \cdot x_2",
        description="critical boron concentration in a PWR",
        target_name="CBORON",
        feature_names=["burnup", "enrichment"],
        n_genes=1,
        coefficients=np.array([1.0]),
        client=client,
    )
    call_kwargs = client.complete.call_args
    prompt = call_kwargs[1].get("prompt") or call_kwargs[0][0]
    assert "critical boron concentration" in prompt
    assert "CBORON" in prompt


def test_interpret_population_passes_feature_names_to_prompt():
    client = _mock_client()
    interpret_population(
        latex=r"x_1",
        description="test domain",
        target_name="y",
        feature_names=["burnup", "enrichment", "boron"],
        n_genes=1,
        coefficients=None,
        client=client,
    )
    call_kwargs = client.complete.call_args
    prompt = call_kwargs[1].get("prompt") or call_kwargs[0][0]
    assert "burnup" in prompt
    assert "enrichment" in prompt
    assert "boron" in prompt


def test_interpret_population_handles_none_coefficients():
    client = _mock_client("Some interpretation.")
    result = interpret_population(
        latex=r"x_1 + x_2",
        description="test",
        target_name="y",
        feature_names=["x1", "x2"],
        n_genes=2,
        coefficients=None,
        client=client,
    )
    assert isinstance(result, str)


def test_interpret_population_handles_empty_latex():
    client = _mock_client("Cannot interpret without expression.")
    result = interpret_population(
        latex="",
        description="test",
        target_name="y",
        feature_names=["x1"],
        n_genes=0,
        coefficients=None,
        client=client,
    )
    assert isinstance(result, str)


def test_population_result_interpret_method():
    """Test interpret() on PopulationResult via duck-typed mock."""
    from symgene.results import PopulationResult

    mock_pop = MagicMock()
    mock_pop.name = "PPF"
    mock_pop.pset.feature_names = ["burnup", "enrichment"]
    mock_pop.best = MagicMock()
    mock_pop.best._combiner.coef_ = np.array([0.6, 0.4])
    mock_pop._hof = []

    mock_evolver = MagicMock()

    pop_result = PopulationResult(population=mock_pop, history=[], evolver=mock_evolver)
    pop_result.to_latex = MagicMock(return_value=r"\sin(burnup) + burnup^2")

    client = _mock_client("The expression shows a sinusoidal burnup dependency.")
    text = pop_result.interpret(
        client=client,
        description="Peak Power Factor in a PWR",
        target_name="PPF",
    )
    assert isinstance(text, str)
    assert len(text) > 0


def test_symgene_result_interpret_returns_dict():
    from symgene.results import SymGeneResult, PopulationResult

    def _make_pop_result(name: str) -> PopulationResult:
        mock_pop = MagicMock()
        mock_pop.name = name
        mock_pop.pset.feature_names = ["x1", "x2"]
        mock_pop.best._combiner.coef_ = np.array([1.0])
        mock_pop._hof = []
        pr = PopulationResult(population=mock_pop, history=[], evolver=MagicMock())
        pr.to_latex = MagicMock(return_value=r"x_1 + x_2")
        return pr

    result = SymGeneResult()
    result["CBORON"] = _make_pop_result("CBORON")
    result["PPF"] = _make_pop_result("PPF")

    client = _mock_client("Interpretation.")
    interpretations = result.interpret(
        client=client,
        descriptions={"CBORON": "boron concentration", "PPF": "power factor"},
    )

    assert set(interpretations.keys()) == {"CBORON", "PPF"}
    assert all(isinstance(v, str) for v in interpretations.values())
