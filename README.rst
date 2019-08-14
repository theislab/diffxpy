|PyPI| |Docs|

.. |PyPI| image:: https://img.shields.io/pypi/v/diffxpy.svg
   :target: https://pypi.org/project/diffxpy
.. |Docs| image:: https://readthedocs.com/projects/diffxpy/badge/?version=latest
   :target: https://diffxpy.readthedocs.io

Fast and scalable differential expression analysis on single-cell RNA-seq data
==============================================================================

diffxpy covers a wide range of differential expression analysis scenarios encountered in single-cell RNA-seq scenarios
and integrates into scanpy_ workflows.
Import diffxpy as `import diffxpy.api as de` to access the following differential expression analysis-related tools:

1. **differential expression analysis** in the module `de.test.*`
2. **gene set enrichment analysis** based on differential expression calls in the module  `de.enrich.*`

Refer to the documentation_ and the tutorials_ for details of these modules.

.. _scanpy: https://github.com/theislab/scanpy
.. _documentation: https://diffxpy.rtfd.io/en/latest
.. _tutorials: https://diffxpy.rtfd.io/en/latest/tutorials.html
