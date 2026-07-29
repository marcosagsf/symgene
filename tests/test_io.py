import pytest
import numpy as np


def _make_result():
    from symgene import PrimitiveSet, Population, SymGeneEvolver
    from symgene.primitives import STANDARD
    pset = PrimitiveSet(n_inputs=2)
    pset.add_from_catalog(STANDARD)
    pop = Population("p", pset, n_genes=2, pop_size=8, n_genes_max=4)
    evolver = SymGeneEvolver(populations=[pop], n_gen=2, verbose=0)
    X = np.random.rand(20, 2)
    y = np.random.rand(20)
    return evolver.fit(X, {"p": y})


def test_save_and_load_result(tmp_path):
    from symgene.io import save_result, load_result
    result = _make_result()
    save_result(result, str(tmp_path / "run1"))
    loaded = load_result(str(tmp_path / "run1"))
    assert "p" in loaded


def test_to_sympy_via_io():
    from symgene.io import to_sympy
    result = _make_result()
    sym = to_sympy(result["p"])
    assert sym is not None


def test_to_latex_via_io():
    from symgene.io import to_latex
    result = _make_result()
    latex = to_latex(result["p"])
    assert isinstance(latex, str)


def test_to_callable_via_io():
    from symgene.io import to_callable
    result = _make_result()
    fn = to_callable(result["p"])
    if fn is not None:
        X = np.random.rand(5, 2)
        y_pred = fn(X)
        assert y_pred.shape == (5,)


def test_to_string_via_io():
    from symgene.io import to_string
    result = _make_result()
    s = to_string(result["p"])
    assert isinstance(s, str)


def test_checkpoint_uses_dill(tmp_path):
    import dill
    from symgene import PrimitiveSet, Population, SymGeneEvolver
    from symgene.primitives import STANDARD
    pset = PrimitiveSet(n_inputs=2)
    pset.add_from_catalog(STANDARD)
    pop = Population("p", pset, n_genes=2, pop_size=8, n_genes_max=4)
    evolver = SymGeneEvolver(
        populations=[pop], n_gen=4, verbose=0,
        checkpoint_dir=str(tmp_path), checkpoint_every=2,
    )
    X = np.random.rand(20, 2)
    y = np.random.rand(20)
    evolver.fit(X, {"p": y})
    ckpt_files = list(tmp_path.glob("*.sgk"))
    assert len(ckpt_files) >= 1
    with open(ckpt_files[0], "rb") as f:
        loaded_evolver = dill.load(f)
    assert hasattr(loaded_evolver, "populations")


def test_load_checkpoint():
    from symgene.io import load_checkpoint
    import dill, tempfile, os
    from symgene import PrimitiveSet, Population, SymGeneEvolver
    from symgene.primitives import STANDARD
    pset = PrimitiveSet(n_inputs=2)
    pset.add_from_catalog(STANDARD)
    pop = Population("p", pset, n_genes=2, pop_size=8, n_genes_max=4)
    evolver = SymGeneEvolver(populations=[pop], n_gen=2, verbose=0)

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.sgk")
        with open(path, "wb") as f:
            dill.dump(evolver, f)
        loaded = load_checkpoint(path)
        assert hasattr(loaded, "populations")
