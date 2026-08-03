# Contributing to SymGene

Thank you for your interest in contributing to SymGene.

## Reporting Bugs

Before opening a bug report, please check whether the issue has already been
reported in the [issue tracker](https://github.com/marcosagsf/symgene/issues).

When filing a new issue, include:
- A minimal reproducible example
- The full error traceback
- Your Python version and SymGene version (`import symgene; print(symgene.__version__)`)
- Operating system and relevant dependency versions

## Suggesting Features

Open a GitHub issue with the label `enhancement`. Describe the use case, the
expected behavior, and how it differs from what the library currently does.

## Submitting Pull Requests

1. Fork the repository and create a branch from `main`.
2. Install the development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
3. Make your changes. Add or update tests in `tests/` to cover the new behavior.
4. Run the test suite before submitting:
   ```bash
   pytest tests/ -v
   ```
5. Open a pull request against `main`. Describe what the PR does and reference
   any related issues.

## Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) conventions.
- Use descriptive variable and function names.
- Keep functions focused on a single responsibility.
- Avoid adding comments that restate what the code does; prefer self-documenting
  names and reserve comments for non-obvious constraints or workarounds.

## Testing

All new features and bug fixes must include tests. Tests live in `tests/` and
are organized to mirror the `symgene/` package structure. The test suite uses
`pytest`. Coverage is measured with `pytest-cov`.

## Code of Conduct

All contributors are expected to follow the project
[Code of Conduct](CODE_OF_CONDUCT.md).
