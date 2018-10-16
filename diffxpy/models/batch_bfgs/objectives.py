import numpy as np
import numpy.random
import scipy.stats


def clip_theta_mu(theta):
    return np.maximum(np.minimum(theta, np.zeros(len(theta)) + 1e8), np.zeros(len(theta)) - 1e8)


def clip_theta_disp(theta):
    return np.maximum(np.minimum(theta, np.zeros(len(theta)) + 1e8), np.zeros(len(theta)) - 1e8)


def nb_glm_linker_mu(theta, X, lib_size):
    return np.exp(np.asarray(np.dot(X, np.asarray(clip_theta_mu(theta)).T)).flatten() + lib_size)


def nb_glm_linker_disp(theta, X):
    return np.asarray(np.exp(np.dot(X, np.asarray(clip_theta_disp(theta)).T))).flatten()


def ll_nb(x, mu, disp):
    # Re-parameterize.
    variance = np.maximum(mu + np.square(mu) / disp, np.zeros(len(mu)) + 1e-8)
    p = 1 - (mu / variance)
    return scipy.stats.nbinom(n=disp, p=1 - p).logpmf(x)


def objective_ll(x, theta_mu, theta_disp, design_loc, design_scale, lib_size):
    return -ll_nb(x=x, mu=nb_glm_linker_mu(theta_mu, design_loc, lib_size),
                  disp=nb_glm_linker_disp(theta_disp, design_scale))


def objective(theta, x, design_loc, design_scale, lib_size, batch_size=100):
    if batch_size is None:
        J = np.sum(objective_ll(x=x,
                                theta_mu=np.asarray(theta)[:design_loc.shape[1]],
                                theta_disp=np.asarray(theta)[design_loc.shape[1]:],
                                design_loc=design_loc,
                                design_scale=design_scale,
                                lib_size=lib_size))
    else:
        batch_idx = numpy.random.randint(low=0, high=x.shape[0], size=(batch_size))
        J = np.sum(objective_ll(x=x[batch_idx],
                                theta_mu=np.asarray(theta)[:design_loc.shape[1]],
                                theta_disp=np.asarray(theta)[design_loc.shape[1]:],
                                design_loc=design_loc[batch_idx, :],
                                design_scale=design_scale[batch_idx, :],
                                lib_size=lib_size[batch_idx]))
    return J
