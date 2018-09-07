.. automodule:: diffxpy.api

Examples
========

Import diffxpy's high-level API as::

   import diffxpy.api as de

Differential expression tests: test
-----------------------------------

Run differential expression tests.

Single tests per gene
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: .

   single/likelihood_ratio_test.ipynb
   single/run_by_partition.ipynb
   single/t_test.ipynb
   single/test_chisq_distribution.ipynb
   single/wald_test.ipynb
   single/wilcoxon_test.ipynb
   
Multiple tests per gene
~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: .

   versus_rest.ipynb
   pairwise.ipynb


Gene set enrichment: enrich
---------------------------

.. autosummary::
   :toctree: .

   enrich.ipynb


Integration with other packages
-------------------------------

.. autosummary::
   :toctree: .

   scanpy_integration.ipynb