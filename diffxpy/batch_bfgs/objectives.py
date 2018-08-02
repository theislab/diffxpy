import numpy as np
import scipy.stats 

def clip_theta_mu(theta):
    np.max([np.min([theta, np.zeros(len(theta))+1e8]), -1e8])

def clip_theta_disp(theta):
    np.max([np.min([theta, np.zeros(len(theta))+1e8]), -1e8])

def nb_glm_linker_mu(theta, X):
    return np.asarray(np.dot(X , np.asmatrix(clip_theta_mu(theta)))).flatten()

def nb_glm_linker_disp(theta, X):
    return np.asarray(np.dot(X , np.asmatrix(clip_theta_disp(theta)))).flatten()

def ll_nb(x, mu, disp):
    # Re-parameterize.
    variance = mu + np.square(mu)/disp
    p = 1 - (mu / variance)
    return scipy.stats.nbinom(n=disp, p=1-p).logpmf(x)

def objective_ll(x, theta_mu, theta_disp, X_mu, X_disp, lib_size):
    return -ll_nb(x=x, mu=nb_glm_linker_mu(theta_mu, X_mu), 
        disp=nb_glm_linker_disp(theta_disp, X_disp))

def objective(theta, x, X_mu, X_disp, lib_size, batch_size=100):
    if batch_size is None
        batch_idx = np.arange(0, x.shape[0])
    else:
        batch_idx = numpy.random.randint(low=0, high=x.shape[0], size=(batch_size))
    return np.sum(objective_ll(x=x[batch], theta_mu=theta[:X_mu.shape[1]], theta_disp=theta[X_mu.shape[1]:], 
        X_mu=X_mu[batch_idx:,], X_disp=X_disp[batch_idx:,], lib_size=lib_size[batch_idx]))