Models
======

Occurrence of estimator objects in diffxpy
------------------------------------------

GLMs and similar models are a main model class for differential expression analysis with Wald and likelihood ratio tests (LRT).
Diffxpy allows the user to choose between different GLMs based on the noise model argument.
The user can select the covariates that are to be modelled based on formulas or by supplying design matrices directly.
Both Wald test (`de.test.wald`) and LRT (`de.test.lrt`) require the fit of GLMs to the given data.
These fits can be extracted from the differential expression test objects that are returned by the `de.test.*` functions:
These objects are called `model_estim` in the case of the Wald test or `full_estim` and `reduced_estim` for the LRT (for full and reduced model).
Similarly, one can use `de.fit.model` to directely produce such an estimator object.

Structure of estimator objects
------------------------------

These estimator objects are the interface between diffxpy and batchglm and can be directly produced with batchglm.
An estimator object contains various attributes that relate to the estimation procedure and a `.model` attribute that contains an executable
(numpy) version of the estimated model. 
The instance of the estimator object contains the raw parameter estimates and functions that compute downstream model characteristics,
such as location and scale parameter estiamtes in a generalized linear model, the equivalent of $\hat{y}$ in a simple feed forward neural network. 
The names of these model attributes depend on the noise model and are listed below

Generalized linear models (GLMs)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The estiamted parameters of the location and scale model are in `estim.model.a_var` (location) and `estim.model.b_var` (scale).
The corresponding parameter names are in `estim.model.loc_names` and `estim.model.scale_names`.
The observation and feature wise location and scale prediction after application of design matrix and inverse linker function are in `estim.model.location` and `estim.model.scale`.

For a negative binomial distribution model, the location model correpsponds to the mean model and the scale model corresponds to the dispersion model.
For a normal distribution model, the location model correpsponds to the mean model and the scale model corresponds to the standard deviation model.
