from scipy import stats


def likelihood_ratio_test(ll_full, ll_reduced, df: int):
    # delta_df = -2 * (ll_H0 - ll_H1) = 2* (ll_H1 - ll_H0)
    delta_df = 2 * (ll_full - ll_reduced)
    return 1 - stats.chi2(df).cdf(delta_df)
