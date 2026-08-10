import importlib.util
import pytest


def llm_deps_installed() -> bool:
    return (
        importlib.util.find_spec("anthropic") is not None
        and importlib.util.find_spec("openai") is not None
    )


requires_llm = pytest.mark.skipif(
    not llm_deps_installed(),
    reason="symgene[llm] not installed — run: pip install symgene[llm]",
)
