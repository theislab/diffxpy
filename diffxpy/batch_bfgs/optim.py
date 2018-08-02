from scipy.optimize import minimize
import numpy as np
import numpy.random 
from multiprocessing import Pool

from objectives import *
from ..base import _Estimation

class Estim_BFGS():
    """ Class that handles multiple parallel starts of parameter estimation on one machine.
    """
    class Estim_BFGS_Model(_Estimation):
        @property
        def num_observations(self) -> int:
            pass
        
        @property
        def num_features(self) -> int:
            pass
        
        @property
        def features(self) -> np.ndarray:
            pass
        
        @property
        def observations(self) -> np.ndarray:
            pass
        
        def probs(self) -> np.ndarray:
            pass
        
        def log_probs(self) -> np.ndarray:
            pass
        
        @property
        def loss(self, **kwargs) -> np.ndarray:
            pass
        
        @property
        def gradient(self, **kwargs) -> np.ndarray:
            pass
        
        @property
        def hessian_diagonal(self, **kwargs) -> np.ndarray:
            pass

    def __init__(self, X, X_mu, X_disp, lib_size, batch_size=100):
        """ Constructor of ManyGenes()
        """
        self.X = X
        self.X_mu = X_mu
        self.X_disp = X_disp
        self.lib_size = lib_size
        self.batch_size = batch_size
        self.__is_sparse = isinstance(X, 'csr-matrix')
        self.res = None


    def __init_mu_theta(self, x, X_mu):
        return np.concatenate([np.mean(x), np.zeros([X_disp.shape[2]-1])])


    def __init_disp_theta(self, x, X_mu):
        return np.zeros([X_disp.shape[2]])


    def __get_gene(self, i):
        if self.__is_sparse:
            return self.X[:,i].todense().flatten()
        else:
            return self.X[:,i].flatten()
        

    def __run_optim(self, x, maxiter=1000, debug=False):
        """ Run single optimisation

        Parameters
        ----------
        """ 
        x0 = np.concatenate(self.__init_mu_theta(x, X_mu),
            self.__init_disp_theta(x, X_disp))
        
        if debug==False:
            try:
                minimize_out = minimize(fun=objective, x0=x0,
                    args=(self.x, self.X_mu, self.X_disp, self.lib_size, self.batch_size),
                    method='BFGS', options={'maxiter':maxiter, 'gtol': 1e-05})
                err = ''
            except Exception as e:
                minimize_out = None
                err = e
                print(e)
        else:
            minimize_out = minimize(fun=objective, x0=x0,
                    args=(self.x, self.X_mu, self.X_disp, self.lib_size, self.batch_size),
                    method='BFGS', options={'maxiter':maxiter, 'gtol': 1e-05})
        
        return minimize_out
        
    def run(self, nproc=1, maxiter=1000, debug=False):
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
        if debug==True:
            self.res = [] # list of FitResults instances.
            for i in range(self.X.shape[1]):
                self.res.append(self.__run_optim(x=self.__get_gene(i), maxiter=maxiter, debug=debug))
                i = i+1
        else:
            with Pool(processes=nproc) as p:
                self.res = p.starmap(self.__run_optim, [(self.__get_gene(i), maxiter, debug) 
                    for i in range(self.X.shape[1])])

    def return_batchglm_formated_model(self):
        model = Estim_BFGS_Model()
        model.num_observations = self.X.shape[0]
        model.num_features = self.X.shape[1]
        model.features = self.X.shape[1]
        model.observations = self.X.shape[0]
        model.probs = np.exp([-x.fun for x in self.res])
        model.log_probs = -np.array([x.fun for x in self.res])
        model.loss = np.array([x.fun for x in self.res])
        model.gradient = np.zeros([self.X.shape[0]])
        model.hessian_diagonal = np.hstack([x.hess.diag() for x in self.res])
        return(model)

    