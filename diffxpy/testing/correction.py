import numpy as np
import statsmodels.stats.multitest


def correct(pvals, method="fdr_bh", alpha=0.05):
    """
    Performs multiple testing corrections available in statsmodels.stats.multitest.multipletests().

    :param pvals: uncorrected p-values. Must be 1-dimensional.
    :param method: Multiple testing correction method.
            Browse available methods in the annotation of statsmodels.stats.multitest.multipletests().
    :param alpha: FWER, family-wise error rate, e.g. 0.1
    """
    # Only correct non-nan p-values.
    qval = np.zeros([pvals.shape[0]]) + np.nan
    qval[np.isnan(pvals) == False] = statsmodels.stats.multitest.multipletests(
        pvals=pvals[np.isnan(pvals) == False],
        alpha=alpha,
        method=method,
        is_sorted=False,
        returnsorted=False
    )[1]

    return qval
