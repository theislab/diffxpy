.. automodule:: diffxpy.api

API
===

Import diffxpy's high-level API as::

   import diffxpy.api as de

Differential expression tests: test
-----------------------------------

Run differential expression tests.
diffxpy distinguishes between single tests and multi tests:
Single tests perform a single hypothesis test for each gene whereas multi tests perform multiple tests per gene.

Single tests per gene
~~~~~~~~~~~~~~~~~~~~~

Single tests per gene are the standard differential expression scenario in which one p-value is computed per gene.
diffxpy provies infrastructure for likelihood ratio tests, Wald tests, t-tests and Wilcoxon tests.

.. autosummary::
   :toctree: .

   test.two_sample
   test.wald
   test.lrt
   test.t_test
   test.wilcoxon

Multiple tests per gene
~~~~~~~~~~~~~~~~~~~~~~~

diffxpy provides infrastructure to perform multiple tests per gene as:

- pairwise: pairwise comparisons across more than two groups (`de.test.pairwise`, e.g. clusters of cells against each other)
- versus_res:t tests of each group against the rest (`de.test.versus_test`, e.g. clusters of cells against the rest)
- partition: mapping a given differential test across each partition of a data set (`de.test.partition`, e.g. performing differential tests for treatment effects by a second experimental covariate or by cluster of cells).

.. autosummary::
   :toctree: .

   test.pairwise
   test.versus_rest
   test.partition


Gene set enrichment: enrich
---------------------------

diffxpy provides infrastructure for gene set enrichment analysis downstream of differential expression analysis.
Specifically, reference gene set annotation data sets can be loaded or created and can be compared to diffxpy objects 
or results from other differential expression tests.


Reference gene sets
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: .

   enrich.RefSets

Enrichment tests
~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: .

   enrich.test
