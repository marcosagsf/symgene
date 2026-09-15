# SymGene

[![Tests](https://github.com/marcosagsf/symgene/actions/workflows/tests.yml/badge.svg)](https://github.com/marcosagsf/symgene/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/symgene.svg)](https://pypi.org/project/symgene/)
[![Python](https://img.shields.io/pypi/pyversions/symgene.svg)](https://pypi.org/project/symgene/)

**Multi-Gene Genetic Programming (MGGP) library for symbolic regression, surrogate-assisted optimization, and LLM-guided evolution.**

SymGene provides a modular, extensible framework for evolving interpretable closed-form mathematical expressions from data. Built on [DEAP](https://deap.readthedocs.io/), it supports multi-population co-evolution, surrogate-assisted black-box optimization with Particle Swarm Optimization (PSO), full symbolic export via [SymPy](https://www.sympy.org/), and optional LLM assistance for primitive selection, expression interpretation, and Genetic Rescue inside the evolutionary loop.

---

## Features

- **Symbolic regression** — evolve closed-form expressions from data using multi-gene representation
- **Multi-population co-evolution** — independent or cooperating populations evolving on different targets simultaneously
- **Surrogate-assisted optimization** — fit a MGGP surrogate model, then minimize with PSO on the learned surface
- **Custom primitive sets** — extend with domain-specific functions, with optional SymPy mapping for symbolic export
- **Symbolic export** — `to_sympy()`, `to_latex()`, `to_callable()` on any fitted model
- **sklearn-like API** — `SymGeneRegressor` for single-population regression; `SymGeneEvolver` for full multi-population control
- **Callbacks** — `EarlyStopping`, `GenerationLogger`, adaptive parameter scheduling per generation
- **Combiners** — Ridge, Lasso, and Linear regression of gene outputs, with polynomial degree support
- **Selection strategies** — Tournament, Rank, Roulette, Lexicase, and Gene-level selection
- **Benchmark suite** — Koza (3), Nguyen (10), and optimization benchmarks (Forrester, Himmelblau, Ackley, Schwefel)
- **LLM assistance** *(optional)* — primitive selection from domain description, expression interpretation, and Genetic Rescue inside the evolutionary loop

---

## Installation

```bash
pip install symgene
```

To enable LLM features (Anthropic and/or OpenAI):

```bash
pip install symgene[llm]
```

To install from source:

```bash
git clone https://github.com/marcosagsf/symgene.git
cd symgene
pip install -e .
```

---

## Quick Start

```python
import numpy as np
from symgene import SymGeneRegressor

# Generate data: f(x) = x³ + x² + x
X = np.random.uniform(-1, 1, (100, 1))
y = X[:, 0]**3 + X[:, 0]**2 + X[:, 0]

regressor = SymGeneRegressor(n_genes=3, pop_size=50, n_gen=100, seed=0)
regressor.fit(X, y)

print(regressor.best_expression_)   # symbolic string
print(regressor.to_latex())         # LaTeX formula
```

---

## Advanced Usage

### Multi-population symbolic regression

Evolve two populations simultaneously on different target functions using `SymGeneEvolver`:

```python
from symgene import PrimitiveSet, Population, SymGeneEvolver
from symgene.primitives import STANDARD
from symgene.fitness import FitnessEvaluator
from symgene.metrics.regression import mse
from symgene.selection import TournamentSelection

pset = PrimitiveSet(n_inputs=1, feature_names=["x"])
pset.add_from_catalog(STANDARD)

pop1 = Population(name="fn1", pset=pset, n_genes=3, pop_size=50,
                  fitness=FitnessEvaluator(metric=mse),
                  selection=TournamentSelection(size=5))
pop2 = Population(name="fn2", pset=pset, n_genes=3, pop_size=50,
                  fitness=FitnessEvaluator(metric=mse),
                  selection=TournamentSelection(size=5))

evolver = SymGeneEvolver(populations=[pop1, pop2], n_gen=200, seed=0)
results = evolver.fit(
    X_train,
    {"fn1": y1_train, "fn2": y2_train},
    X_val=X_val,
    y_val={"fn1": y1_val, "fn2": y2_val},
)
print(results["fn1"].best_expression_)
```

### Surrogate-assisted optimization

Fit a MGGP surrogate on sampled data, then use PSO to find the minimum:

```python
from symgene import SymGeneRegressor
from symgene.benchmarks import himmelblau_2d
from symgene.optimization import PSOOptimizer

bench = himmelblau_2d()
# ... generate X_train, y_train via Latin Hypercube Sampling ...

regressor = SymGeneRegressor(
    n_genes=4, pop_size=60, n_gen=100,
    feature_names=["x1", "x2"], seed=0,
)
regressor.fit(X_train, y_train, X_val=X_val, y_val=y_val)

optimizer = PSOOptimizer(n_particles=50, n_iter=300)
result = optimizer.optimize(
    lambda x: float(regressor.predict(x.reshape(1, -1))[0]),
    bounds=bench.bounds,
    seed=0,
)
print(f"Found: x={result.x_best}  f={result.f_best:.4f}")
```

### Custom primitives with symbolic export

Register domain-specific functions with a SymPy counterpart for full symbolic export:

```python
import numpy as np
import sympy as sp
from symgene import PrimitiveSet
from symgene.primitives import ARITHMETIC

pset = PrimitiveSet(n_inputs=2, feature_names=["x1", "x2"])
pset.add_from_catalog(ARITHMETIC + ["sin", "cos"])
pset.add_custom(
    fn=lambda x: float(x * np.sin(np.sqrt(abs(x)))),
    arity=1,
    name="xsinqrt",
    sympy_fn=lambda x: x * sp.sin(sp.sqrt(sp.Abs(x))),
)

# ... fit model with SymGeneEvolver ...
print(regressor.to_latex())     # renders custom primitive in LaTeX
fn = regressor.to_callable()    # standalone Python function, no MGGP dependency
```

### LLM-assisted evolution

Requires `pip install symgene[llm]` and an Anthropic or OpenAI API key.

**Phase 1 — LLM selects primitives from a domain description:**

```python
from symgene.llm import LLMClient, InsufficientContextError

client = LLMClient(provider="anthropic", model="claude-haiku-4-5-20251001")

try:
    pset = PrimitiveSet.from_description(
        description="axial power distribution in a PWR with burnup and enrichment gradients",
        client=client,
        n_inputs=3,
        feature_names=["burnup", "enrichment", "boron"],
    )
except InsufficientContextError as e:
    print(e.llm_message)   # LLM asks for more context
```

**Phase 2 — LLM interprets the evolved expression:**

```python
from symgene.llm import LLMContext

result = evolver.fit(X, {"PPF": y_ppf})
print(result["PPF"].interpret(client, description="peak power factor in a PWR"))

# Build a concept library from the run
ctx = LLMContext.from_result(result["PPF"], client, description="PWR power distribution")
ctx.evolve(client, n_concepts=3)
print(ctx.concepts)
```

**Phase 3 — Genetic Rescue inside the evolutionary loop:**

```python
evolver = SymGeneEvolver(
    populations=[pop],
    n_gen=300,
    llm_rescue=True,
    llm_client=client,
    llm_context={"PPF": ctx},
    llm_rescue_trigger="stagnation",   # fires when best fitness stops improving
    llm_rescue_level="gene",           # replace one gene per rescued individual
    llm_rescue_fraction=0.1,           # rescue worst 10% of population
    llm_stagnation_patience=20,
)
result = evolver.fit(X, {"PPF": y_ppf})
```

If the LLM API becomes unreachable, a warning is printed and evolution continues normally without LLM assistance.

---

## Examples

The `examples/` directory contains five self-contained scripts covering the main use cases.
Rich visualisations and full execution outputs are available in the companion `notebooks/`.

| Script | Benchmark | Key features demonstrated |
|---|---|---|
| `01_mggp_forrester_koza.py` | Forrester 1D + Koza-1 | `SymGeneEvolver`, multi-population, custom primitive, callbacks |
| `02_surrogate_himmelblau.py` | Himmelblau 2D | Surrogate + PSO, `combiner="lasso"`, `RouletteSelection`, `regression_degree=2` |
| `03_surrogate_ackley.py` | Ackley 2D | Surrogate + PSO, `combiner="ridge"`, `TournamentSelection` |
| `04_mggp_nguyen10_doublewell.py` | Nguyen-10 + Double Well | `RankSelection`, `missing_vars_penalty`, `schedule`, `EarlyStopping` |
| `05_schwefel_custom_pset.py` | Schwefel 2D | `ARITHMETIC` preset, `add_custom` + `sympy_fn`, `to_latex()`, `to_callable()` |
| `06_llm_primitive_selection.py` *(requires `[llm]`)* | Nguyen-10 | `PrimitiveSet.from_description()`, `InsufficientContextError`, LLM vs STANDARD pset comparison |
| `07_llm_interpret_and_concepts.py` *(requires `[llm]`)* | Forrester 1D | `LLMContext.from_result()`, `ctx.evolve()`, `PopulationResult.interpret()` |
| `08_llm_genetic_rescue.py` *(requires `[llm]`)* | Dittus-Boelter (heat transfer) | Full LLM pipeline: Phase 1 + Genetic Rescue + Phase 2 |

Run any example after installation:

```bash
python examples/01_mggp_forrester_koza.py
```

---

## API Overview

| Class / Function | Description |
|---|---|
| `SymGeneRegressor` | High-level sklearn-like regressor for single-population MGGP |
| `SymGeneEvolver` | Low-level driver for multi-population co-evolution |
| `Population` | Encapsulates one population: genes, operators, fitness, selection |
| `PrimitiveSet` | Manages mathematical primitives and terminal nodes |
| `PrimitiveSet.from_description()` | Build a primitive set from a domain description via LLM *(requires `[llm]`)* |
| `FitnessEvaluator` | Combines a metric with optional complexity/diversity penalties |
| `PSOOptimizer` | Particle Swarm Optimizer for black-box minimization |
| `SymGeneResult` | Fitted model: `predict()`, `to_sympy()`, `to_latex()`, `to_callable()`, `interpret()` |
| `EarlyStopping` | Stop evolution when a monitored metric plateaus |
| `GenerationLogger` | Print generation statistics at a chosen interval |

**LLM module** (`from symgene.llm import ...` — requires `pip install symgene[llm]`)

| Class / Function | Description |
|---|---|
| `LLMClient` | Provider-agnostic LLM wrapper (Anthropic / OpenAI) |
| `LLMContext` | Library of natural-language concepts guiding LLM-assisted evolution |
| `InsufficientContextError` | Raised when description is too vague for primitive selection |
| `suggest_primitives()` | Ask LLM to select primitives for a given domain |
| `abstract_concepts()` | Extract mathematical patterns from good/bad expressions |
| `evolve_concepts()` | Refine and extend a concept library using LLM |
| `rescue_worst()` | Genetic Rescue operator — rehabilitate worst individuals via LLM |

### Primitive presets

| Preset | Contents |
|---|---|
| `ARITHMETIC` | `add`, `sub`, `mul`, `div` |
| `STANDARD` | `ARITHMETIC` + power, trigonometric, exponential, hyperbolic, activation, statistical |
| `EXTENDED` | `STANDARD` + additional hyperbolic, activation, and conditional functions |
| `ALL` | `EXTENDED` + `if_greater` |

### Benchmark functions

```python
from symgene.benchmarks import (
    forrester_1d, himmelblau_2d, ackley_2d, schwefel_2d,  # optimization
    koza1, koza2, koza3,                                    # Koza suite
    nguyen1, nguyen5, nguyen7, nguyen9, nguyen10,           # Nguyen suite
)
bench = forrester_1d()
# bench.fn(x), bench.bounds, bench.x_opt, bench.f_opt, bench.formula, bench.name
```

---

## Running Tests

```bash
pip install symgene[dev]
pytest tests/ -v --cov=symgene
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Citation

If you use SymGene in your research, please cite:

```bibtex
@article{symgene2026,
  title   = {SymGene: A Multi-Gene Genetic Programming Library for Symbolic Regression
             and Surrogate-Assisted Optimization},
  author  = {Filho, Marcos A. G. S.},
  journal = {SoftwareX},
  year    = {2026},
}
```
