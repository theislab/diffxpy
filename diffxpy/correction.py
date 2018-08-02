import statsmodels.stats.multitest

def correct(pvals, method="fdr_bh"):
    """
    Performs multiple testing corrections available in statsmodels.stats.multitest.multipletests().
    

    :param method: Multiple testing correction method.
            Browse available methods in the annotation of statsmodels.stats.multitest.multipletests().
    """
    qval = statsmodels.stats.multitest.multipletests(pvals=pvals, 
    	alpha=0.05, method=method, is_sorted=False, returnsorted=False)[1]
    return qval