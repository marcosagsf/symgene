---
title: 'SymGene: A Multi-Gene Genetic Programming Library for Symbolic Regression and Surrogate-Assisted Optimization'
tags:
  - Python
  - genetic programming
  - symbolic regression
  - surrogate optimization
  - interpretable machine learning
  - evolutionary computation
authors:
  - name: Marcos A.G.S. Filho
    orcid: 0009-0002-8510-9509
    affiliation: 1
affiliations:
  - name: Department of Nuclear Engineering, Federal University of Rio de Janeiro (UFRJ), Brazil
    index: 1
date: 3 August 2026
bibliography: paper.bib
---

# Summary

SymGene is an open-source Python library for multi-gene genetic programming (MGGP)
applied to symbolic regression and surrogate-assisted optimization. MGGP represents
each model as a weighted linear combination of evolved tree-based expressions, called
genes, fitted by ridge or lasso regression. This structure produces interpretable,
closed-form models that can be exported as SymPy expressions, LaTeX strings, or
standalone Python callables requiring no library dependency at evaluation time.

The library provides two interfaces: a high-level `SymGeneRegressor` following
scikit-learn conventions, and a low-level `SymGeneEvolver` supporting simultaneous
multi-population co-evolution. Each population can be configured with independent
selection strategies, fitness evaluators, primitive sets, and adaptive parameter
schedules. An integrated Particle Swarm Optimizer (PSO) enables surrogate-assisted
optimization workflows, in which MGGP learns an analytical surface from sampled data
and PSO minimizes it at negligible additional computational cost.

# Statement of Need

Interpretable models are essential in scientific and engineering domains where the
learned expression must be inspected, validated against physical theory, or deployed
without software dependencies. Black-box models such as neural networks and gradient
boosting achieve strong predictive performance but do not produce expressions amenable
to this kind of analysis.

Genetic programming (GP) is the principal framework for symbolic regression: the
automated search for mathematical expressions fitting observed data. Among its
variants, MGGP offers a practical balance between expressiveness and interpretability.
By combining multiple gene trees through linear regression, MGGP models are richer
than single-tree GP while remaining fully interpretable as closed-form equations.

The need for accessible MGGP implementations arises directly from applied research.
In nuclear engineering, MGGP has been used to generate analytical expressions for
critical boron concentration and power peak factor as functions of reactor reload
parameters, replacing iterative numerical calculations with fast, auditable formulas
[@filho2025]. Similar needs arise in geotechnical parameter estimation, structural
reliability analysis, and control system identification. In all these domains,
practitioners require not just a predictive model but a transferable equation.

SymGene was designed to meet this need. It provides native MGGP with multi-population
support in a single Python package, without requiring users to assemble the capability
from lower-level evolutionary framework primitives.

# State of the Field

Several libraries address symbolic regression in Python. gplearn [@stephens2015]
provides a scikit-learn compatible single-tree GP regressor but does not support
multi-gene representation or surrogate-assisted optimization. PySR [@cranmer2023]
is a high-performance library with a Julia-based search engine and excellent
scalability, but its algorithmic approach differs from classical MGGP and does not
support simultaneous evolution of independent populations on distinct target functions.
DEAP [@fortin2012] is a general evolutionary computation framework from which MGGP
workflows can be constructed, but doing so requires substantial user-level boilerplate
code. GPTIPS [@searson2010] implemented MGGP in MATLAB but is no longer actively
maintained and unavailable in Python.

SymGene occupies the gap between these options: a Python-native MGGP library with
multi-population co-evolution, PSO surrogate optimization, and full symbolic export,
built on DEAP [@fortin2012] and compatible with scikit-learn.

# Software Design

SymGene is organized into independent submodules covering primitives, selection,
operators, combiners, metrics, optimization, surrogate modeling, callbacks, and
benchmarks.

`SymGeneRegressor` wraps a single population and exposes `fit()` and `predict()`
methods following the scikit-learn estimator interface. `SymGeneEvolver` enables
multi-population co-evolution: users construct `Population` objects, each holding
its own configuration, and pass them together to the evolver, which coordinates
training across all populations and returns a dictionary of `SymGeneResult` objects.

Primitive sets are managed by `PrimitiveSet`. Four presets are provided: `ARITHMETIC`,
`STANDARD`, `EXTENDED`, and `ALL`. Custom functions are registered through `add_custom`,
which accepts a Python callable and an optional `sympy_fn` argument. When `sympy_fn`
is provided, the symbolic export pipeline uses it during tree traversal, enabling
exact symbolic representation of domain-specific functions without string parsing.

Gene outputs are aggregated by a combiner (ridge, lasso, or linear regression). The
`regression_degree` parameter expands gene features polynomially before fitting,
enabling degree-2 interactions between genes without manually constructing product
terms.

Selection strategies include `TournamentSelection`, `RankSelection`,
`RouletteSelection`, and `LexicaseSelection`. Fitness is computed by
`FitnessEvaluator`, which combines a base regression metric with optional penalties
for tree complexity and missing input variables. Adaptive parameter scheduling is
defined through generation-indexed checkpoints, with the evolver applying linear
interpolation between them.

Callbacks implement an `on_generation_end` hook. Built-in callbacks include
`EarlyStopping` and `GenerationLogger`. Users subclass `Callback` to implement
custom logic.

Surrogate-assisted optimization pairs MGGP with `PSOOptimizer`. MGGP learns an
analytical surrogate on Latin Hypercube Sampled data; PSO then minimizes the surrogate
callable. This workflow reduces the number of true objective evaluations required for
black-box optimization.

Symbolic export is provided through `SymGeneResult.to_sympy()`, `to_latex()`, and
`to_callable()`. The `to_callable()` method compiles the expression into a standalone
Python function with no SymGene dependency.

A benchmark suite covering Koza [@koza1992], Nguyen, and multimodal optimization
functions (Forrester, Himmelblau, Ackley, Schwefel) is included for reproducibility.
Five illustrative examples and companion Jupyter notebooks demonstrate the main
library features across regression and surrogate optimization tasks.

# Research Impact

SymGene directly extends the methodology described in @filho2025, where multi-gene
genetic programming was applied to nuclear reactor fuel reload calculations at the
Angra II power plant. The library packages this methodology into a reusable,
documented form so that researchers in other engineering and scientific domains can
apply multi-population MGGP without reimplementing the core evolutionary machinery.

The surrogate-assisted optimization workflow is relevant to any field with expensive
function evaluations, where fitting a fast analytical surrogate before optimizing
reduces computational burden. The symbolic export pipeline addresses a practical
limitation common to genetic programming implementations: the final model often
exists as an internal tree structure and must be reconstructed for deployment. The
`to_callable()` method eliminates this dependency entirely.

# AI Usage Disclosure

Generative AI assistance (Claude, Anthropic) was used in drafting and revising
portions of this manuscript and in preparing code documentation. All technical
content, results, and design decisions were reviewed and validated by the author.

# Acknowledgements

M.A.G.S. Filho acknowledges financial support from the Carlos Chagas Filho Foundation
for Research Support of the State of Rio de Janeiro (FAPERJ) through a graduate 
scholarship.

# References
