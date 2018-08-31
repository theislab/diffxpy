
# Fast and scalable differential expression analysis on single-cell RNA-seq data
diffxpy covers a wide range of differential expression analysis scenarios encountered in single-cell RNA-seq scenarios. The core ideas of diffxpy is to speed up the model fitting time, which is the run-time bottle-neck of differential expression analysis on models that require maximum likelihood estimators and that do not have closed form solutions thereof. This model fitting is performed in a separate package, batchglm. diffxpy exposes wrapper funtions to the user which perform the model fitting and the differential expression test. Advanced users can chose the between various hypothesis tests and can vary the model fitting hyperparameters. Fitting can be performed both on CPUs and on GPUs and can be parallelized.

# Installation
1. Install [batchglm](https://github.com/theislab/batchglm).
2. Clone the GitHub repository of diffxpy.
3. cd into the clone.
4. pip install -e .

# Worklows and API
Diffxpy supports workflows related to differential expression analysis. The functionalities of the workflows are structured in the API. To access these workflows, import diffxpy as follows `import diffxpy.api as de`. Currently, the following analytic strategies are implemented:

1. differential expression analysis in `de.test.*`
2. gene set enrichment analysis based on differential expression calls in `de.enrich.*`

The aforementioned workflows can be concatenated in pipelines and results can be shared via diffxpy data structures as explained in the individual functions and in the examples.

# Examples
We provide usage example cases (vignettes) in the `examples/` directory. The diffxpy API distinguishes two cases: Single-tests and Mult-tests. Single-test wrappers perform a single tests per gene, such as a difference between two groups via a Wald test, a log-likelihood ratio test, a t-test or a Wilcoxon rank sum test or a log-likelihood ratio test for a more complex question. A multi-test wrapper performs multiple tests per gene: pairwise or 1-versus-rest tests. Both are a series of two-sample tests for each gene which can for example be used to explore the differences between louvain groups on the level of each gene. Pairwise tests perform one test for the difference between each pair of groups of cells for each gene (which can be significantly sped up if the test mode 'z-test' is used). 1-versus-rest tests test for the difference between each group and the other groups together for each gene. Examples for additional workflows like gene set enrichment analysis are also provided.

# Building the documentation
The documentation is maintained in the `docs/` directory.

The built documentation will be saved in `build/docs`. 
 
1. Make sure sphinx is installed (install via pip for example). 
2. `cd docs/`
3. `make html`
