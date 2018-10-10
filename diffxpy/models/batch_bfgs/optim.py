import logging

from scipy.optimize import minimize
from scipy.sparse import csr_matrix
import numpy as np
from numpy.linalg import pinv
import numpy.random
from multiprocessing import Pool
import xarray as xr

from .objectives import *

logger = logging.getLogger(__name__)


class Estim_BFGS_Model():

    def __init__(self, Estim_BFGS, nproc):
        self._num_observations = Estim_BFGS.X.shape[0]
        self._num_features = Estim_BFGS.X.shape[1]
        self._features = Estim_BFGS.feature_names
        self._observations = Estim_BFGS.X.shape[0]
        self._design_loc = Estim_BFGS.design_loc
        self._design_scale = Estim_BFGS.design_scale
        self._loss = xr.DataArray(Estim_BFGS.full_loss(nproc))
        self._log_probs = -self._loss
        self._probs = np.exp(self._log_probs)
        self._mles = xr.DataArray(np.transpose(Estim_BFGS.mles()))
        self._gradient = xr.DataArray(np.zeros([Estim_BFGS.X.shape[1]]))
        self._fisher_inv = xr.DataArray(Estim_BFGS.fisher_inv)
        self._idx_loc = np.arange(0, Estim_BFGS.design_loc.shape[1])
        self._idx_scale = np.arange(Estim_BFGS.design_loc.shape[1],
                                    Estim_BFGS.design_loc.shape[1] + Estim_BFGS.design_scale.shape[1])
        self._error_codes = Estim_BFGS.error_codes()
        self._niter = Estim_BFGS.niter()
        self._estim_bfgs = Estim_BFGS

    @property
    def num_observations(self) -> int:
        return self._num_observations

    @property
    def num_features(self) -> int:
        return self._num_features

    @property
    def features(self) -> np.ndarray:
        return self._features

    @property
    def observations(self) -> np.ndarray:
        return self._observations

    @property
    def design_loc(self, **kwargs):
        return self._design_loc

    @property
    def design_scale(self, **kwargs):
        return self._design_scale

    @property
    def probs(self):
        return self._probs

    # @property
    def log_probs(self):
        return self._log_probs

    @property
    def loss(self, **kwargs):
        return self._loss

    @property
    def gradient(self, **kwargs):
        return self._gradient

    @property
    def mles(self, **kwargs):
        return self._mles

    @property
    def par_link_loc(self, **kwargs):
        return self._mles[self._idx_loc, :]

    @property
    def par_link_scale(self, **kwargs):
        return self._mles[self._idx_scale, :]

    def link_loc(self, x):
        return np.log(x)

    @property
    def fisher_inv(self):
        return self._fisher_inv

    @property
    def error_codes(self, **kwargs):
        return self._error_codes


class Estim_BFGS():
    """ Class that handles multiple parallel starts of parameter estimation on one machine.
    """

    def __init__(self, X, design_loc, design_scale, lib_size=0, batch_size=None, feature_names=None):
        """ Constructor of ManyGenes()
        """
        if np.size(lib_size):
            lib_size = np.broadcast_to(lib_size, X.shape[0])

        if batch_size is not None:
            logger.warning("Using BFGS with batching is currently not supported!")

        self.X = X
        self.design_loc = np.asarray(design_loc)
        self.design_scale = np.asarray(design_scale)
        self.lib_size = lib_size
        self.batch_size = batch_size
        self.feature_names = feature_names
        self.__is_sparse = isinstance(X, csr_matrix)
        self.res = None

    def __init_mu_theta(self, x):
        if self.design_loc.shape[1] > 1:
            return np.concatenate([[np.log(np.mean(x) + 1e-08)], np.zeros([self.design_loc.shape[1] - 1])])
            # return np.zeros([self.design_loc.shape[1]])
        else:
            return [np.log(np.mean(x) + 1e-08)]

    def __init_disp_theta(self, x):
        if self.design_scale.shape[1] > 1:
            return np.zeros([self.design_scale.shape[1]])
        else:
            return [0]

    def get_gene(self, i):
        """

        Has to be public so that it can be passed via starmap.
        """
        if self.__is_sparse:
            return np.asarray(self.X[:, i].data.todense())
        else:
            return np.asarray(self.X[:, i].data)

    def run_optim(self, x, maxiter=10000, debug=False):
        """ Run single optimisation.

        Has to be public so that it can be passed via starmap.

        Parameters
        ----------
        """
        x0 = np.concatenate([self.__init_mu_theta(x), self.__init_disp_theta(x)])

        if debug:
            minimize_out = minimize(
                fun=objective,
                x0=x0,
                args=(x, self.design_loc, self.design_scale, self.lib_size, self.batch_size),
                method='BFGS',
                options={'maxiter': maxiter, 'gtol': 1e-05}
            )
        else:
            try:
                minimize_out = minimize(
                    fun=objective,
                    x0=x0,
                    args=(x, self.design_loc, self.design_scale, self.lib_size, self.batch_size),
                    method='BFGS',
                    options={'maxiter': maxiter, 'gtol': 1e-05}
                )
                err = ''
            except Exception as e:
                minimize_out = None
                err = e
                print(e)

        return minimize_out

    def run(self, nproc=1, maxiter=10000, debug=False):
        """ Run multiple optimisation starts
        
        Parameters
        ----------
        maxiter : int
            Maximum number of iterations in parameter estimation.
        nstarts : int 
            Number of starts to perform in multi-start optimization.
        nproc : int
            Number of processes to use in parallelization.
        """
        if debug == True:
            self.res = []  # list of FitResults instances.
            for i in range(self.X.shape[1]):
                self.res.append(self.run_optim(x=self.get_gene(i), maxiter=maxiter, debug=debug))
        else:
            with Pool(processes=nproc) as p:
                self.res = p.starmap(
                    self.run_optim,
                    [(self.get_gene(i), maxiter, False) for i in range(self.X.shape[1])]
                )

    def full_loss(self, nproc=1):
        with Pool(processes=nproc) as p:
            loss = np.asarray(p.starmap(
                objective,
                [(x.x, self.get_gene(i), self.design_loc, self.design_scale, self.lib_size, None)
                 for i, x in enumerate(self.res)]
            ))
        return np.asarray(loss)

    def mles(self):
        return np.vstack([x.x for x in self.res])

    def fim_diags(self):
        return np.vstack([x.hess_inv.diagonal() for x in self.res])

    @property
    def fisher_inv(self):
        return np.stack([x.hess_inv for x in self.res])

    def niter(self):
        return np.array([x.nit for x in self.res])

    def error_codes(self):
        return np.array([x.status for x in self.res])

    def return_batchglm_formated_model(self, nproc=1):
        model = Estim_BFGS_Model(self, nproc)
        return (model)
