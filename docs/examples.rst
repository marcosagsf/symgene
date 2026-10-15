Examples
========

The ``examples/`` directory contains eight self-contained scripts covering the
main use cases, from basic symbolic regression to the full LLM-assisted
pipeline. Companion Jupyter notebooks with figures and full execution output
live under ``notebooks/``.

Run any script after installation:

.. code-block:: bash

   python examples/01_mggp_forrester_koza.py

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Script
     - Benchmark
     - Features demonstrated
   * - `01_mggp_forrester_koza.py <https://github.com/marcosagsf/symgene/blob/main/examples/01_mggp_forrester_koza.py>`_
     - Forrester 1D + Koza-1
     - ``SymGeneEvolver``, multi-population, custom primitive, callbacks
   * - `02_surrogate_himmelblau.py <https://github.com/marcosagsf/symgene/blob/main/examples/02_surrogate_himmelblau.py>`_
     - Himmelblau 2D
     - Surrogate + PSO, Lasso combiner, degree-2 regression
   * - `03_surrogate_ackley.py <https://github.com/marcosagsf/symgene/blob/main/examples/03_surrogate_ackley.py>`_
     - Ackley 2D
     - Surrogate + PSO, Ridge combiner, tournament selection
   * - `04_mggp_nguyen10_doublewell.py <https://github.com/marcosagsf/symgene/blob/main/examples/04_mggp_nguyen10_doublewell.py>`_
     - Nguyen-10 + Double Well
     - Rank selection, missing-variable penalty, scheduling, early stopping
   * - `05_schwefel_custom_pset.py <https://github.com/marcosagsf/symgene/blob/main/examples/05_schwefel_custom_pset.py>`_
     - Schwefel 2D
     - Custom primitives with ``sympy_fn``, ``to_latex()``, ``to_callable()``
   * - `06_llm_primitive_selection.py <https://github.com/marcosagsf/symgene/blob/main/examples/06_llm_primitive_selection.py>`_
     - Nguyen-10
     - ``PrimitiveSet.from_description()`` *(requires* ``[llm]`` *)*
   * - `07_llm_interpret_and_concepts.py <https://github.com/marcosagsf/symgene/blob/main/examples/07_llm_interpret_and_concepts.py>`_
     - Forrester 1D
     - ``LLMContext``, concept library, expression interpretation *(requires* ``[llm]`` *)*
   * - `08_llm_genetic_rescue.py <https://github.com/marcosagsf/symgene/blob/main/examples/08_llm_genetic_rescue.py>`_
     - Dittus-Boelter heat transfer
     - Full LLM pipeline: primitive selection + Genetic Rescue + interpretation *(requires* ``[llm]`` *)*

The notebooks are browsable directly on GitHub:
`notebooks/ <https://github.com/marcosagsf/symgene/tree/main/notebooks>`_.
