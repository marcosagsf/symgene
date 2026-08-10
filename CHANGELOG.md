# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

**Dittus-Boelter heat transfer benchmark**
- `dittus_boelter()` — symbolic regression benchmark for the convective heat transfer correlation `Nu = 0.023·Re^0.8·Pr^0.4`; log-uniform LHS sampling over Re ∈ [10⁴, 10⁶] and Pr ∈ [0.7, 160]; returns `BenchmarkData` with `feature_names=["Re", "Pr"]`
- `BenchmarkData` NamedTuple extended with optional `feature_names` field (backward-compatible default `None`)
- 10 new tests in `tests/benchmarks/test_heat_transfer.py`

**LLM example scripts** *(require `pip install symgene[llm]` + API key)*
- `examples/06_llm_primitive_selection.py` — Phase 1: compares LLM-selected vs STANDARD primitive set on Nguyen-10; demonstrates `PrimitiveSet.from_description()` and `InsufficientContextError`
- `examples/07_llm_interpret_and_concepts.py` — Phase 2: fits Forrester 1D then builds and evolves a concept library with `LLMContext`; interprets the best expression with `PopulationResult.interpret()`
- `examples/08_llm_genetic_rescue.py` — Full pipeline (Phases 1+2+3): primitive selection → Genetic Rescue during training → concept extraction → physical interpretation, all on the Dittus-Boelter benchmark

---

## [0.2.0] — 2026-08-10

### Added

**LLM-Assisted Evolution (`symgene[llm]` optional extra)**

*Phase 1 — LLM-guided primitive set selection*
- `LLMClient` — provider-agnostic LLM wrapper supporting Anthropic and OpenAI SDKs (`symgene/llm/client.py`)
- `PrimitiveSet.from_description()` — classmethod that selects domain-relevant primitives from a natural-language description via LLM
- `suggest_primitives()` — standalone function wrapping the LLM primitive selection prompt
- `InsufficientContextError` — raised when the LLM determines the description is too vague; `llm_message` attribute contains the LLM's explanation of what additional context is needed
- Dual response format protocol: FORMAT A (JSON primitive list) / FORMAT B (needs_more_info object)
- `pip install symgene[llm]` optional dependency group (`anthropic>=0.34`, `openai>=1.0`)
- Lazy loading via `__getattr__` in `symgene/llm/__init__.py`; base SymGene has zero LLM import cost

*Phase 2 — Post-evolution LLM analysis*
- `PopulationResult.interpret()` — asks LLM for physical interpretation of the best evolved expression
- `SymGeneResult.interpret()` — multi-population wrapper returning `{pop_name: interpretation}` dict
- `LLMContext` — Library of Concepts dataclass accumulating natural-language patterns across evolution runs
- `LLMContext.from_result()` — builds context by abstracting concepts from best/worst HOF expressions
- `LLMContext.evolve()` — refines and extends the concept library using the LLM; returns `self` for chaining
- `abstract_concepts()` — identifies mathematical patterns distinguishing good from bad expressions
- `evolve_concepts()` — generates refined concept combinations from existing library

*Phase 3 — Genetic Rescue (inside the evolutionary loop)*
- `rescue_worst()` — LLM operator that rehabilitates the worst `rescue_fraction` individuals using structure from the best individual + LLMContext concepts
- Gene-level rescue: replaces one random gene per rescued individual
- Individual-level rescue: replaces all genes of each rescued individual
- `gp.PrimitiveTree.from_string()` parser: prefix-notation string → `SGGene`, with tree_max validation and markdown/numbering cleanup
- `_call_llm_with_retry()`: retries on parse/format errors up to `llm_max_retries`; API exceptions propagate to caller
- `SymGeneEvolver` LLM parameters: `llm_client`, `llm_context`, `llm_rescue`, `llm_rescue_fraction`, `llm_rescue_trigger` (`"stagnation"` / `"random"` / `"both"`), `llm_rescue_level` (`"gene"` / `"individual"`), `llm_stagnation_patience`, `llm_rescue_prob`, `llm_max_retries`
- Per-population stagnation detection; counter resets after rescue fires
- API error handling: prints warning, sets `_llm_disabled = True`, run continues normally without LLM
- 18 new unit tests in `tests/llm/test_rescue.py`

*General LLM infrastructure*
- 36 unit tests across `tests/llm/` (primitive_suggest, interpret, context, rescue) — all using mocked clients, no real API calls
- `tests/llm/conftest.py` with `requires_llm` skip marker for tests that need real SDK installation

**Paper and documentation**
- JOSS paper (`paper.md`) with summary, statement of need, software design, and research impact sections
- `CONTRIBUTING.md` with contribution guidelines
- `CODE_OF_CONDUCT.md` (Contributor Covenant)
- Author ORCID, co-authors, DOI, and acknowledgements in JOSS paper
- Citations for SymPy and scikit-learn in `paper.bib`

---

## [0.1.0] — 2026-07-29

### Added

**Core primitives**
- `Squash` — sigmoidal explosion control for GP tree outputs (`symgene/primitives/squash.py`)
- Arithmetic primitive factory: `add`, `sub`, `mul`, `div`, `abs`
- Power primitive factory: `square`, `cube`, `sqrt`, `cbrt`, `inv`, `pow`
- Trigonometric primitive factory: `sin`, `cos`, `tan`, `atan`, `asin`, `acos`
- Exponential primitive factory: `exp`, `log`, `log2`, `log10`
- Hyperbolic primitive factory: `tanh`, `sinh`, `cosh`, `atanh`
- Activation primitive factory: `sigmoid`, `relu`, `softplus`, `softsign`, `swish`, `elu`, `gaussian`, `sinc`
- Statistical primitive factory: `mean2/3/4`, `max2/3/4`, `min2/3/4`, `harmonic_mean2`, `geometric_mean2`
- Logical primitive factory: `if_positive`, `if_greater`, `step`
- Primitives catalog with preset lists `STANDARD`, `EXTENDED`, `ALL` and `get_catalog()`

**PrimitiveSet**
- `PrimitiveSet` — fluent API for building DEAP primitive sets with custom primitives, ephemerals, and squash control

**Combiners**
- `RidgeCombiner` — Ridge regression with cross-validated alpha and optional polynomial degree
- `LassoCombiner` — Lasso regression with cross-validated alpha
- `LinearCombiner` — ordinary least squares
- All combiners expose `gene_weights_` for importance-based gene selection

**Metrics**
- Regression metrics: `mse`, `rmse`, `mae`, `r2`, `nrmse`, `mape`
- Penalty functions: `complexity_penalty`, `missing_vars_penalty`
- Diversity metrics: `genotypic_diversity`, `phenotypic_diversity`

**Fitness**
- `FitnessEvaluator` — pluggable metric + ordered penalty list

**Selection**
- `TournamentSelection` — k-tournament with configurable size
- `RouletteSelection` — fitness-proportionate selection
- `RankSelection` — linear rank-based selection with configurable selection pressure
- `LexicaseSelection` — epsilon-lexicase selection for continuous targets
- `select_genes_by_weight` — gene selection weighted by Ridge coefficients

**Population**
- `Population` — DEAP-backed multi-gene individual management with `evaluate()`, `apply_schedule()`, per-case error tracking (`_case_errors`), and configurable combiner/selection/fitness

**Operators**
- Intra-population crossover (subtree crossover between genes of the same individual)
- Inter-population crossover (gene exchange between individuals of different populations)
- High-order and low-order subtree mutation
- Island model migration with `ring`, `full`, and `best_to_worst` topologies

**Evolver**
- `SymGeneEvolver` — single and multi-population evolutionary loop with validation, checkpointing (dill), early stopping signal support, and callback hooks (`on_train_begin`, `on_generation_end`, `on_train_end`)

**Callbacks**
- `Callback` — base class with no-op hooks
- `GenerationLogger` — stdout and file logging every N generations with metric filtering
- `EarlyStopping` — stops evolution when monitored metric stagnates (patience, min_delta, mode)
- `ParameterScheduler` — sets object attributes at specified generation thresholds
- `ReduceOnPlateau` — reduces an attribute by a factor when metric stops improving

**Results**
- `SymGeneResult` — dict-like container for multi-population results
- `PopulationResult` — exposes `predict()`, `to_sympy()`, `to_latex()`, `to_callable()`, `to_string()`, Pareto front, `save()` / `load()`, `plot_expression_tree()`, `plot_gene_weights()`

**Regressor**
- `SymGeneRegressor` — scikit-learn-style single-population wrapper (`fit`, `predict`)

**IO**
- `save_checkpoint` / `load_checkpoint` — evolver serialization using dill
- `save_result` / `load_result` — SymGeneResult round-trip persistence
- Standalone export functions: `to_sympy`, `to_latex`, `to_callable`, `to_string`

**Visualization**
- `plot_tree` — pure matplotlib GP expression tree (no networkx dependency)
- `plot_individual_trees` — side-by-side gene trees for a multi-gene individual
- `plot_n_genes_over_time` — gene count across generations
- `plot_gene_weights` — Ridge coefficient bar chart per gene
- `plot_complexity_distribution` — histogram of total node counts across population

**Benchmarks**
- Koza suite: `koza1`, `koza2`, `koza3`
- Nguyen suite: `nguyen1` through `nguyen10`
- Optimization functions: `forrester`, `himmelblau`, `ackley`

**Optimizer**
- `PSOOptimizer` — Particle Swarm Optimization with configurable swarm size, inertia, and cognitive/social coefficients

**Surrogate**
- `SymGeneSurrogate` — MGGP-based surrogate model with LHS sampling and PSO optimization pipeline

**Examples**
- `examples/nuclear_reactor/` — two-population MGGP demo with 108 synthetic inputs
- `examples/medicine_bmd/` — `SymGeneRegressor` demo on synthetic BMD dataset
- `examples/surrogate_forrester/` — Forrester function surrogate with MGGP
- `examples/surrogate_himmelblau/` — Himmelblau PSO optimization via surrogate
- `examples/surrogate_ackley/` — Ackley function surrogate comparison

### Fixed
- `RankSelection` probability formula was inverted; rank 1 (best) now correctly receives highest selection probability
- `GenerationLogger` file handle leak when training raised an exception; `on_train_end` now called in `finally` block
- Insufficient training samples in Forrester and Ackley example scripts causing poor surrogate fits

### Changed
- All Portuguese comments and docstrings translated to English
- Example scripts refactored to distribute hyperparameter variety across three scripts (removed nuclear/BMD hard dependency)

[Unreleased]: https://github.com/marcosagsf/symgene/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/marcosagsf/symgene/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/marcosagsf/symgene/releases/tag/v0.1.0
