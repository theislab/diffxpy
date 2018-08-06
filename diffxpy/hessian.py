import numpy as np
from scipy.special import polygamma
import numpy.linalg


def hes_nb_glm_mean_block(
        x: np.ndarray,
        mu: np.ndarray,
        disp: np.ndarray,
        design_loc: np.ndarray,
        design_scale: np.ndarray,
        i: int,
        j: int
):
    """ Compute entry of hessian in mean model block for a given gene.

    Sum the following across cells:
    $h_{ij} = -x^{m_i}*x^{m_j}*mu*(x/disp)/(1+mu(disp))^2$
    Make sure that only element wise operations happen here!
    Do not simplify design matrix elements: they are only 0 or 1 for discrete
    groups but continuous if space, time, pseudotime or spline basis covariates
    are supplied!

    :param x: np.ndarray (cells,)
        Observations for a given gene.
    :param mu: np.ndarray (cells,)
        Estimated mean parameters across cells for a given gene.
    :param mu: np.ndarray (cells,)
        Estimated dispersion parameters across cells for a given gene.
    :param design_loc: np.ndarray, matrix, xarray (cells, #parameters location model)
        Design matrix of location model.
    :param design_scale: np.ndarray, matrix, xarray (cells, #parameters shape model)
        Design matrix of shape model.
    :param i: int
        Index of first dimension in fisher information matrix which is to be computed.
    :param j: int
        Index of second dimension in fisher information matrix which is to be computed
    :return: float
        Entry of fisher information matrix in mean model block at position (i,j)
    """
    
    h_ij = -np.asarray(design_loc[:, i]) * np.asarray(design_loc[:, j]) * mu * x / disp / np.square(1 + mu / disp)
    return np.sum(h_ij)


def hes_nb_glm_disp_block(
        x: np.ndarray,
        mu: np.ndarray,
        disp: np.ndarray,
        design_loc: np.ndarray,
        design_scale: np.ndarray,
        i: int,
        j: int
):
    """ Compute entry of hessian in dispersion model block for a given gene.

    Sum the following across cells:
    $$
    h_{ij} =
        disp * x^{m_i} * x^{m_j} * [psi_0(disp+x)
        + psi_0(disp)
        - mu/(disp+mu)^2 * (disp+x)
        +(mu-disp) / (disp+mu)
        + log(disp)
        + 1 - log(disp+mu)]
        + disp * psi_1(disp+x)
        + disp * psi_1(disp)
    $$
    
    Make sure that only element wise operations happen here!
    Do not simplify design matrix elements: they are only 0 or 1 for discrete
    groups but continuous if space, time, pseudotime or spline basis covariates
    are supplied!
    
    :param x: np.ndarray (cells,)
        Observations for a given gene.
    :param mu: np.ndarray (cells,)
        Estimated mean parameters across cells for a given gene.
    :param mu: np.ndarray (cells,)
        Estimated dispersion parameters across cells for a given gene.
    :param design_loc: np.ndarray, matrix, xarray (cells, #parameters location model)
        Design matrix of location model.
    :param design_scale: np.ndarray, matrix, xarray (cells, #parameters shape model)
        Design matrix of shape model.
    :param i: int
        Index of first dimension in fisher information matrix which is to be computed.
    :param j: int
        Index of second dimension in fisher information matrix which is to be computed

    :return: float
        Entry of fisher information matrix in dispersion model block at position (i,j)
    """
    h_ij = (
            disp * np.asarray(design_loc[:, i]) * np.asarray(design_loc[:, j]) * polygamma(n=0, x=disp + x)
            + polygamma(n=0, x=disp)
            - mu / np.square(disp + mu) * (disp + x)
            + (mu - disp) / (disp + mu)
            + np.log(disp)
            + 1 - np.log(disp + mu)
            + disp * polygamma(n=1, x=disp + x)
            + disp * polygamma(n=1, x=disp)
    )
    return np.sum(h_ij)


def hes_nb_glm_meandisp_block(
        x: np.ndarray,
        mu: np.ndarray,
        disp: np.ndarray,
        design_loc: np.ndarray,
        design_scale: np.ndarray,
        i: int,
        j: int
):
    """ Compute entry of hessian in mean-dispersion model block for a given gene.

    Sum the following across cells:
    Need to multiply by -1 here??????
    $h_{ij} = mu*x^{m_i}*x^{m_j}*(x-mu)/(1+mu/disp)^2$
    
    Make sure that only element wise operations happen here!
    Do not simplify design matrix elements: they are only 0 or 1 for discrete
    groups but continuous if space, time, pseudotime or spline basis covariates
    are supplied!
    
    :param x: np.ndarray (cells,)
        Observations for a given gene.
    :param mu: np.ndarray (cells,)
        Estimated mean parameters across cells for a given gene.
    :param mu: np.ndarray (cells,)
        Estimated dispersion parameters across cells for a given gene.
    :param design_loc: np.ndarray, matrix, xarray (cells, #parameters location model)
        Design matrix of location model.
    :param design_scale: np.ndarray, matrix, xarray (cells, #parameters shape model)
        Design matrix of shape model.
    :param i: int
        Index of first dimension in fisher information matrix which is to be computed.
    :param j: int
        Index of second dimension in fisher information matrix which is to be computed

    :return: float
        Entry of fisher information matrix in mean-dispersion model block at position (i,j)
    """
    h_ij = disp * np.asarray(design_loc[:, i]) * np.asarray(design_loc[:, j]) * (x - mu) / np.square(1 + mu / disp)
    return np.sum(h_ij)


def hes_nb_glm_bygene(
        x: np.ndarray,
        mu: np.ndarray,
        disp: np.ndarray,
        design_loc: np.ndarray,
        design_scale: np.ndarray,
):
    """ Compute hessian for a given gene.

    :param x: np.ndarray (cells,)
        Observations for a given gene.
    :param mu: np.ndarray (cells,)
        Estimated mean parameters across cells for a given gene.
    :param mu: np.ndarray (cells,)
        Estimated dispersion parameters across cells for a given gene.
    :param design_loc: np.ndarray, matrix, xarray (cells, #parameters location model)
        Design matrix of location model.
    :param design_scale: np.ndarray, matrix, xarray (cells, #parameters shape model)
        Design matrix of shape model.
    
    :return: np.ndarray (#parameters location model + #parameters shape model, #parameters location model + #parameters shape model)
        Fisher information matrix.
    """
    n_par_loc = design_loc.shape[1]
    n_par_scale = design_scale.shape[1]
    n_par = n_par_loc + n_par_scale
    hes = np.zeros([n_par, n_par])
    # Add in elements by block:
    # Mean model block:
    for i in np.arange(0, n_par_loc):
        for j in np.arange(i, n_par_loc):  # Block is on the diagonal and symmtric.
            hes[i, j] = hes_nb_glm_mean_block(x=x, mu=mu, disp=disp, design_loc=design_loc, design_scale=design_scale,
                                              i=i, j=j)
            hes[j, i] = hes[i, j]
    # Dispersion model block:
    for i in np.arange(0, n_par_scale):
        for j in np.arange(i, n_par_scale):  # Block is on the diagonal and symmtric.
            hes[n_par_loc + i, n_par_loc + j] = hes_nb_glm_disp_block(x=x, mu=mu, disp=disp, design_loc=design_loc,
                                                                      design_scale=design_scale, i=i, j=j)
            hes[n_par_loc + j, n_par_loc + i] = hes[n_par_loc + i, n_par_loc + j]
    # Mean-dispersion model block:
    for i in np.arange(0, n_par_loc):
        for j in np.arange(0, n_par_scale):  # Duplicate block across diagonal but block itself is not symmetric!
            hes[i, n_par_loc + j] = hes_nb_glm_meandisp_block(x=x, mu=mu, disp=disp, design_loc=design_loc,
                                                              design_scale=design_scale, i=i, j=j)
            hes[n_par_loc + j, i] = hes[i, n_par_loc + j]
    return (hes)


def theta_covar_bygene(
        x: np.ndarray,
        mu: np.ndarray,
        disp: np.ndarray,
        design_loc: np.ndarray,
        design_scale: np.ndarray,
):
    """ Compute model coefficient covariance matrix for a given gene.

    Based on the hessian matrix via fisher information matrix (fim).
    covar = inv(fim) = inv(-hess)

    :param x: np.ndarray (cells,)
        Observations for a given gene.
    :param mu: np.ndarray (cells,)
        Estimated mean parameters across cells for a given gene.
    :param mu: np.ndarray (cells,)
        Estimated dispersion parameters across cells for a given gene.
    :param design_loc: np.ndarray, matrix, xarray (cells, #parameters location model)
        Design matrix of location model.
    :param design_scale: np.ndarray, matrix, xarray (cells, #parameters shape model)
        Design matrix of shape model.
    
    :return: np.ndarray (#parameters location model + #parameters shape model, #parameters location model + #parameters shape model)
        Model coefficient covariance matrix.
    """
    hes = hes_nb_glm_bygene(x=x, mu=mu, disp=disp, design_loc=design_loc, design_scale=design_scale)
    return numpy.linalg.pinv(-hes)


def theta_sd_bygene(
        x: np.ndarray,
        mu: np.ndarray,
        disp: np.ndarray,
        design_loc: np.ndarray,
        design_scale: np.ndarray,
):
    """ Compute model coefficient standard deviation vector for a given gene.

    Based on the hessian matrix via fisher information matrix (fim).
    covar = inv(fim) = inv(-hess)
    var = diagonal of covar

    :param x: np.ndarray (cells,)
        Observations for a given gene.
    :param mu: np.ndarray (cells,)
        Estimated mean parameters across cells for a given gene.
    :param disp: np.ndarray (cells,)
        Estimated dispersion parameters across cells for a given gene.
    :param design_loc: np.ndarray, matrix, xarray (cells, #parameters location model)
        Design matrix of location model.
    :param design_scale: np.ndarray, matrix, xarray (cells, #parameters shape model)
        Design matrix of shape model.
    
    :return: np.ndarray (#parameters location model + #parameters shape model,)
        Model coefficient standard deviation vector.
    """
    
    hes = hes_nb_glm_bygene(x=x, mu=mu, disp=disp, design_loc=design_loc, design_scale=design_scale)
    return np.sqrt(numpy.linalg.pinv(-hes).diagonal())
