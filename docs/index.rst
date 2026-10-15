SymGene
=======

**SymGene** is a Multi-Gene Genetic Programming (MGGP) library for symbolic
regression, surrogate-assisted optimization, and LLM-guided evolution. It
evolves interpretable closed-form expressions from data and exports them to
SymPy, LaTeX, or a dependency-free Python callable.

The library is built on `DEAP <https://deap.readthedocs.io/>`_ and follows
scikit-learn conventions, so a fitted model plugs into ``cross_val_score``,
``GridSearchCV``, and ``clone``.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Usage

   examples

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api/modules

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
