import numpy as np
import scipy.stats
from typing import Union


def likelihood_ratio_test(
        ll_full: np.ndarray,
        ll_reduced: np.ndarray,
        df_full: int,
        df_reduced: int
):
    """
    Perform log-likihood ratio test based on already fitted models.

    The reduced model has to be nested within the full model
    for the deviance to be chi-square distributed under the null
    hypothesis: The p-values are incorrect if the models are not nested.

    :param ll_full: np.array (genes)
        Likelihood of full model for each gene.
    :param ll_reduced:  np.array (genes)
        Likelihood of reduced model for each gene.
    :param df_full: int
        Degrees of freedom (number of parameters) of full model for each gene.
    :param df_reduced:  int
        Degrees of freedom (number of parameters) of reduced model for each gene.
    """
    if ll_full.shape[0] != ll_full.shape[0]:
        raise ValueError('stats.likelihood_ratio_test(): ll_full and ll_red have to contain the same number of genes')
    # Compute the difference in degrees of freedom.
    delta_df = df_full - df_reduced
    # Compute the deviance test statistic.
    delta_dev = 2 * (ll_full - ll_reduced)
    # Compute the p-values based on the deviance and its expection based on the chi-square distribution.
    pvals = 1 - scipy.stats.chi2(delta_df).cdf(delta_dev)
    return pvals


def wilcoxon_test(
        x0: np.ndarray,
        x1: np.ndarray,
):
    """
    Perform Wilcoxon rank sum test (Mann-Whitney U test) along second axis
    (for each gene).

    The Wilcoxon rank sum test is a non-parameteric test
    to compare two groups of observations.

    :param x0: np.array (observations x genes)
        Observations in first group by gene
    :param x1:  np.array (observations x genes)
        Observations in second group by gene.
    """
    axis = 1
    if np.any(np.ndim(x0) != np.ndim(x1)):
        raise ValueError('stats.wilcoxon(): number of dimensions is not allowed to differ between x0 and x1!')
    # Reshape into 2D array if only one test is performed.
    if np.ndim(x0) == 1:
        x0 = x0.reshape([x0.shape[0], 1])
        x1 = x1.reshape([x1.shape[0], 1])
    if np.any(x0.shape[axis] != x1.shape[axis]):
        raise ValueError(
            'stats.wilcoxon(): the first axis (number of tests) is not allowed to differ between x0 and x1!')

    pvals = np.asarray([
        scipy.stats.mannwhitneyu(
            x=x0[:, i].flatten(),
            y=x1[:, i].flatten(),
            alternative='two-sided'
        ).pvalue for i in range(x0.shape[1])
    ])
    return pvals


def t_test_raw(
        x0,
        x1,
):
    """
    Perform two-sided t-test allowing for unequal group variances (Welch's t-test) on raw data
    along second axis (for each gene).

    The t-test assumes normally distributed data. This function computes
    the necessary statistics and calls t_test_moments().

    :param x0: np.array (observations x genes)
        Observations in first group by gene
    :param x1:  np.array (observations x genes)
        Observations in second group by gene.
    """
    axis = 1
    if x0.shape[1] != x1.shape[1]:
        raise ValueError('stats.t_test_raw(): x0 and x1 have to contain the same number of genes')
    # Reshape into 2D array if only one test is performed.
    if np.ndim(x0) == 1:
        x0 = x0.reshape([x0.shape[0], 1])
        x1 = x1.reshape([x1.shape[0], 1])
    if np.any(x0.shape[axis] != x1.shape[axis]):
        raise ValueError(
            'stats.wilcoxon(): the first axis (number of tests) is not allowed to differ between x0 and x1!')

    mu0 = np.asarray(np.mean(x0, axis=0)).flatten()
    var0 = np.asarray(np.var(x0, axis=0)).flatten()
    mu1 = np.asarray(np.mean(x1, axis=0)).flatten()
    var1 = np.asarray(np.var(x1, axis=0)).flatten()
    n0 = x0.shape[0]
    n1 = x1.shape[0]

    pval = t_test_moments(mu0=mu0, mu1=mu1, var0=var0, var1=var1, n0=n0, n1=n1)
    return pval


def t_test_moments(
        mu0: np.ndarray,
        mu1: np.ndarray,
        var0: np.ndarray,
        var1: np.ndarray,
        n0: int,
        n1: int
):
    """
    Perform two-sided t-test allowing for unequal group variances (Welch's t-test)
    moments of distribution of data.

    The t-test assumes normally distributed data.

    :param mu0: np.array (genes)
        Mean expression by gene of first group.
    :param mu1 np.array (genes)
        Mean expression by gene of second group.
    :param var0: np.array (genes)
        Variance of expression by gene of first group.
    :param var1 np.array (genes)
        Variance of expression by gene of second group.
    :param n0: np.array (genes)
        Number of observations in first group.
    :param n1 np.array (genes)
        Number of observations in second group.
    """
    if len(mu0) != len(mu1):
        raise ValueError('stats.t_test_moments(): mu and mu1 have to contain the same number of entries')
    if len(var0) != len(var1):
        raise ValueError('stats.t_test_moments(): mu and mu1 have to contain the same number of entries')

    s_delta = np.sqrt((var0 / n0) + (var1 / n1))
    np.clip(
        s_delta,
        a_min=np.nextafter(0, np.inf, dtype=s_delta.dtype),
        a_max=np.nextafter(np.inf, 0, dtype=s_delta.dtype),
        out=s_delta
    )

    t_statistic = np.abs((mu0 - mu1) / s_delta)

    divisor = (
            (np.square(var0 / n0) / (n0 - 1)) +
            (np.square(var1 / n1) / (n1 - 1))
    )
    np.clip(
        divisor,
        a_min=np.nextafter(0, np.inf, dtype=divisor.dtype),
        a_max=np.nextafter(np.inf, 0, dtype=divisor.dtype),
        out=divisor
    )

    with np.errstate(over='ignore'):
        df = np.square((var0 / n0) + (var1 / n1)) / divisor
    np.clip(
        df,
        a_min=np.nextafter(0, np.inf, dtype=df.dtype),
        a_max=np.nextafter(np.inf, 0, dtype=df.dtype),
        out=df
    )

    pval = 2 * (1 - scipy.stats.t(df).cdf(t_statistic))
    return pval


def wald_test(
        theta_mle: np.ndarray,
        theta_sd: np.ndarray,
        theta0: Union[int, np.ndarray] = 0
):
    """
    Perform single coefficient Wald test.

    The Wald test unit_test whether a given coefficient deviates significantly
    from the supplied reference value, based on the standard deviation
    of the posterior of the parameter estimate. In the context of
    generalized linear nodels, this standard deviation
    is typically obtained via the hessian at the maximum likelihood
    estimator, which is an approximation of the fisher information matrix,
    based on which the parameter covariance matrix can be calculated,
    which has the standard deviation of each the distribution of each
    parameter on its diagonal.

    :param theta_mle: np.array (genes)
        Maximum likelihood estimator of given parameter by gene.
    :param theta_sd:  np.array (genes)
        Standard deviation of maximum likelihood estimator of given parameter by gene.
    :param theta0: float
        Reference parameter values against which coefficient is tested.
    """
    if np.size(theta0) == 1:
        theta0 = np.broadcast_to(theta0, theta_mle.shape)

    if theta_mle.shape[0] != theta_sd.shape[0]:
        raise ValueError('stats.wald_test(): theta_mle and theta_sd have to contain the same number of entries')
    if theta0.shape[0] > 1:
        if theta_mle.shape[0] != theta0.shape[0]:
            raise ValueError('stats.wald_test(): theta_mle and theta0 have to contain the same number of entries')

    wald_statistic = np.abs(np.divide(theta_mle - theta0, theta_sd))
    pvals = 2 * (1 - scipy.stats.norm(loc=0, scale=1).cdf(wald_statistic))  # two-tailed test
    return pvals


def wald_test_chisq(
        theta_mle: np.ndarray,
        theta_invcovar: np.ndarray,
        theta0: Union[int, np.ndarray] = 0
):
    """
    Perform single coefficient Wald test.

    The Wald test unit_test whether a given coefficient deviates significantly
    from the supplied reference value, based on the standard deviation
    of the posterior of the parameter estimate. In the context of
    generalized linear nodels, this standard deviation
    is typically obtained via the hessian at the maximum likelihood
    estimator, which is an approximation of the fisher information matrix,
    based on which the parameter covariance matrix can be calculated,
    which has the standard deviation of each the distribution of each
    parameter on its diagonal.

    :param theta_mle: np.array (par x genes)
        Maximum likelihood estimator of given parameter by gene.
    :param theta_invcovar:  np.array (genes x par x par)
        Inverse of the covariance matrix of the parameters in theta_mle by gene.
        This is the negative hessian or the inverse of the 
        observed fisher information matrix.
    :param theta0: float
        Reference parameter values against which coefficient is tested.
    """
    if np.size(theta0) == 1:
        theta0 = np.broadcast_to(theta0, theta_mle.shape)

    if theta_mle.shape[0] != theta_invcovar.shape[1]:
        raise ValueError(
            'stats.wald_test(): theta_mle and theta_invcovar have to contain the same number of parameters')
    if theta_mle.shape[1] != theta_invcovar.shape[0]:
        raise ValueError('stats.wald_test(): theta_mle and theta_invcovar have to contain the same number of genes')
    if theta_invcovar.shape[1] != theta_invcovar.shape[2]:
        raise ValueError('stats.wald_test(): the first two dimensions of theta_invcovar have to be of the same size')
    if theta0.shape[0] > 1:
        if theta_mle.shape[0] != theta0.shape[0]:
            raise ValueError('stats.wald_test(): theta_mle and theta0 have to contain the same number of entries')
        if theta_mle.shape[1] != theta0.shape[1]:
            raise ValueError('stats.wald_test(): theta_mle and theta0 have to contain the same number of entries')

    theta_diff = theta_mle - theta0
    wald_statistic = np.array([
        np.matmul(
            np.matmul(
                theta_diff[:, [i]].T,
                theta_invcovar[i, :, :]
            ),
            theta_diff[:, [i]]
        )
        for i in range(theta_diff.shape[1])
    ]).flatten()
    pvals = 1 - scipy.stats.chi2(theta_mle.shape[0]).cdf(wald_statistic)
    return pvals


def two_coef_z_test(
        theta_mle0: np.ndarray,
        theta_mle1: np.ndarray,
        theta_sd0: np.ndarray,
        theta_sd1: np.ndarray
):
    """
    Perform z-test to compare two coefficients.

    The Wald test tests whether a given coefficient deviates significantly
    from the other coefficient, based on their standard deviations
    of the posterior of the parameter estimates. In the context of
    generalized linear models, this standard deviation
    is typically obtained via the hessian at the maximum likelihood
    estimator, which is an approximation of the fisher information matrix,
    based on which the parameter covariance matrix can be calculated,
    which has the standard deviation of each the distribution of each
    parameter on its diagonal.

    :param theta_mle0: np.array (genes)
        Maximum likelihood estimator of first parameter by gene.
    :param theta_mle1: np.array (genes)
        Maximum likelihood estimator of second parameter by gene.
    :param theta_sd0:  np.array (genes)
        Standard deviation of maximum likelihood estimator of first parameter by gene.
    :param theta_sd1:  np.array (genes)
        Standard deviation of maximum likelihood estimator of second parameter by gene.
    """
    if theta_mle0.shape[0] != theta_mle1.shape[0]:
        raise ValueError(
            'stats.two_coef_z_test(): theta_mle0 and theta_mle1 have to contain the same number of entries')
    if theta_sd0.shape[0] != theta_sd1.shape[0]:
        raise ValueError('stats.two_coef_z_test(): theta_sd0 and theta_sd1 have to contain the same number of entries')
    if theta_mle0.shape[0] != theta_sd0.shape[0]:
        raise ValueError('stats.two_coef_z_test(): theta_mle0 and theta_sd0 have to contain the same number of entries')

    z_statistic = np.abs((theta_mle0 - theta_mle1) / np.sqrt(np.square(theta_sd0) + np.square(theta_sd1)))
    pvals = 2 * (1 - scipy.stats.norm(loc=0, scale=1).cdf(z_statistic))  # two-tailed test
    return pvals


def hypergeom_test(
        intersections: np.ndarray,
        enquiry: int,
        references: np.ndarray,
        background: int
) -> np.ndarray:
    """ Run a hypergeometric test (gene set enrichment).

    The scipy docs have a nice explanation of the hypergeometric ditribution and its parameters.
    This function wraps scipy.stats.hypergeom() and can compare multiple reference (enquiry)
    sets against one target set.

    :param intersections: np.ndarray
        Array with number of overlaps of reference sets with enquiry set.
    :param enquiry: np.ndarray
        Size of each enquiry set to be tested.
    :param references: int
        Array with size of reference sets.
    :param background: int
        Size of background set.
    """
    pvals = np.array([1 - scipy.stats.hypergeom(M=background, n=references[i], N=enquiry).cdf(x - 1) for i, x in
                      enumerate(intersections)])
    return (pvals)
