Training
=========

Parameter estimation in diffxpy
-------------------------------

diffxpy performs parameter estimation for generalized linear models (GLMs) with batchglm.
GLMs are necessary for Wald tests and liklihood ratio-tests, not for t-tests and Wilcoxon rank-sum tests.
batchglm exploits closed form maximum likelihood estimators in GLMs where possible, but often numerical parameter estimation is necessary.
Parameters for GLMs can be estiamted with iteratively weighted least squares (IWLS) (exponential family GLMs)
or via standard methods for maximum likelihood estimation which are based on local approximations of the objective function (e.g. gradient decent).
The latter cover a larger range of variance models and are applicable for all noise models and were therefore chosen for batchglm.
However, these methods often come with hyper-parameters (such as learning rates).
Differential expression frameworks often hide the training from the user, 
diffxpy exposes training details to the user so that training can be monitored and hyperparameters optimized.
To reduce the coding effor and technical knowledge necessary for this, we expose core hyper-parameters within "training-strategies".


Training strategies
-------------------

Training strategies give the user to opportunity to change optimzer defaults such as the 
optimizer algorithm, learning rates, optimizer schedules (multiple optimizers) and convergence criteria.
Please post issues on GitHub if you notice that your model does not converge with the default optimizer.