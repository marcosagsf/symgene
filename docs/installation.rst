Installation
============

From PyPI
---------

.. code-block:: bash

   pip install symgene

LLM features (primitive selection, interpretation, Genetic Rescue) require the
optional ``llm`` extra, which pulls in the Anthropic, OpenAI, and Google GenAI
SDKs:

.. code-block:: bash

   pip install symgene[llm]

From source
-----------

.. code-block:: bash

   git clone https://github.com/marcosagsf/symgene.git
   cd symgene
   pip install -e .

To build the documentation or run the test suite locally, install the matching
extras:

.. code-block:: bash

   pip install -e ".[dev]"    # pytest, pytest-cov
   pip install -e ".[docs]"   # sphinx, furo

Requirements
------------

SymGene supports Python 3.10, 3.11, and 3.12. Core dependencies (NumPy,
scikit-learn, SymPy, matplotlib, pandas, DEAP, dill, joblib) are installed
automatically.
