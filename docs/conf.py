"""Sphinx configuration for the SymGene documentation."""
import os
import sys
from datetime import date

# Make the package importable for autodoc when building from a source checkout.
sys.path.insert(0, os.path.abspath(".."))

import symgene  # noqa: E402

project = "SymGene"
author = "Marcos A. G. S. Filho"
copyright = f"{date.today().year}, {author}"
release = symgene.__version__
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",      # NumPy-style docstrings
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
]

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

# Optional LLM SDKs are not installed on the docs builder; mock them so that
# autodoc can import symgene.llm without the extras.
autodoc_mock_imports = ["anthropic", "openai", "google"]

napoleon_numpy_docstring = True
napoleon_google_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "sympy": ("https://docs.sympy.org/latest/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = f"SymGene {release}"
html_static_path = ["_static"]
