Quickstart
==========

Fit a symbolic model with the scikit-learn-style :class:`~symgene.SymGeneRegressor`:

.. code-block:: python

   import numpy as np
   from symgene import SymGeneRegressor

   # Target: f(x) = x**3 + x**2 + x
   X = np.random.uniform(-1, 1, (100, 1))
   y = X[:, 0] ** 3 + X[:, 0] ** 2 + X[:, 0]

   regressor = SymGeneRegressor(n_genes=3, pop_size=50, n_gen=100, seed=0)
   regressor.fit(X, y)

   print(regressor.best_expression_)   # symbolic string
   print(regressor.to_latex())         # LaTeX formula

The fitted model exposes :meth:`~symgene.results.PopulationResult.to_sympy`,
:meth:`~symgene.results.PopulationResult.to_latex`, and
:meth:`~symgene.results.PopulationResult.to_callable` for exporting the evolved
expression.

For multi-population co-evolution, surrogate-assisted optimization with PSO,
custom primitives, and LLM-assisted evolution, see :doc:`examples`.
