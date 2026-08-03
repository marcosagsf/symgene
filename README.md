# SymGene

**Multi-Gene Genetic Programming (MGGP) library for symbolic regression and surrogate-assisted optimization.**

SymGene provides a modular, extensible framework for evolving interpretable closed-form mathematical expressions from data. Built on [DEAP](https://deap.readthedocs.io/), it supports multi-population co-evolution, surrogate-assisted black-box optimization with Particle Swarm Optimization (PSO), and full symbolic export via [SymPy](https://www.sympy.org/).

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

---

## Installation

```bash
pip install symgene
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
| `FitnessEvaluator` | Combines a metric with optional complexity/diversity penalties |
| `PSOOptimizer` | Particle Swarm Optimizer for black-box minimization |
| `SymGeneResult` | Fitted model: `predict()`, `to_sympy()`, `to_latex()`, `to_callable()` |
| `EarlyStopping` | Stop evolution when a monitored metric plateaus |
| `GenerationLogger` | Print generation statistics at a chosen interval |

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
  author  = {Filho, Marcos},
  journal = {SoftwareX},
  year    = {2026},
}
```
