
# Fast and scalable differential expression analysis on single-cell RNA-seq data

diffxpy covers a wide range of differential expression analysis scenarios encountered in single-cell RNA-seq scenarios. The core ideas of diffxpy is to speed up the model fitting time, which is the run-time bottle-neck of differential expression analysis on models that require maximum likelihood estimators and that do not have closed form solutions thereof. This model fitting is performed in a separate package, batchglm. diffxpy exposes wrapper funtions to the user which perform the model fitting and the differential expression test. Advanced users can chose the between various hypothesis tests and can vary the model fitting hyperparameters. Fitting can be performed both on CPUs and on GPUs and can be parallelized.

# Installation

1. Install tensorflow, refer to https://github.com/theislab/batchglm for details.
2. Clone the GitHub repository of diffxpy.
3. cd into the clone.
4. pip install -e .

# Building the documentation
The documentation is maintained in the `docs/` directory.

The built documentation will be saved in `build/docs`. 
 
1. Make sure sphinx is installed (install via pip for example). 
2. `cd docs/`
3. `make html`
