import logging
import unittest
import numpy as np
import scipy.stats as stats

import diffxpy.api as de


class TestStats(unittest.TestCase):
    
    def test_lrt(self, df: int = 3, n: int = 1000):
        """
        Test if de.stats.likelihood_ratio_test() generates a uniform p-value distribution
        if it is given test statistics sampled from the null model. Returns the p-value
        of the two-side Kolmgorov-Smirnov test for equality of the observed 
        p-value distriubution and a uniform distribution.

        :param n: Number of tests to run.
        :param df: Difference in degrees of freedom between null and alternative model.
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)
        
        # Draw chi-square distributed deviance which is the statistic 
        # distributed under the null hypothesis:
        # dev = 2 * (ll_full - ll_reduced)
        dev = np.random.chisquare(df=df, size=n)
        
        # Set ll_full, ll_red and df_full and df_red so that the correct
        # deviance is computed within likelihood_ratio_test().
        ll_full = dev / 2
        ll_red = np.zeros_like(ll_full)
        
        # Compute p-value distribution under null model.
        pvals = de.stats.likelihood_ratio_test(ll_full=ll_full, ll_reduced=ll_red, df_full=df, df_reduced=0)
        
        # Compare p-value distribution under null model against uniform distribution.
        pval_h0 = stats.kstest(pvals, 'uniform').pvalue

        logging.getLogger("diffxpy").info('KS-test pvalue for null model match of likelihood_ratio_test(): %f' % pval_h0)
        assert pval_h0 > 0.05, "KS-Test failed: pval_h0 is <= 0.05!"

        return True
    
    def test_wald(self, n: int = 1000):
        """
        Test if de.stats.wald() generates a uniform p-value distribution
        if it is given test statistics sampled from the null model. Returns the p-value
        of the two-side Kolmgorov-Smirnov test for equality of the observed 
        p-value distriubution and a uniform distribution.

        :param n: Number of tests to run.
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)
        
        # Draw standard normal distributed estimate which is sampled
        # from the parameter posterior under the null model:
        mles = np.random.normal(loc=0, scale=1, size=n)
        sd = np.zeros([n]) + 1
        
        # Compute p-value distribution under null model.
        pvals = de.stats.wald_test(theta_mle=mles, theta_sd=sd, theta0=0)
        
        # Compare p-value distribution under null model against uniform distribution.
        pval_h0 = stats.kstest(pvals, 'uniform').pvalue

        logging.getLogger("diffxpy").info('KS-test pvalue for null model match of wald(): %f' % pval_h0)
        assert pval_h0 > 0.05, "KS-Test failed: pval_h0 is <= 0.05!"

        return True
    
    def test_z_test(self, n: int = 1000):
        """
        Test if de.stats.two_coef_z_test() generates a uniform p-value distribution
        if it is given test statistics sampled from the null model. Returns the p-value
        of the two-side Kolmgorov-Smirnov test for equality of the observed 
        p-value distriubution and a uniform distribution.

        :param n: Number of tests to run.
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)
        
        # Draw parameter posteriors for each test:
        theta_mles = np.random.normal(loc=0, scale=1, size=n)
        theta_sds = np.exp(np.random.normal(loc=0, scale=0.5, size=n))
        
        # Draw two estimates from each posterior:
        theta_mle0 = np.random.normal(loc=theta_mles, scale=theta_sds)
        theta_mle1 = np.random.normal(loc=theta_mles, scale=theta_sds)
        
        # Compute p-value distribution under null model.
        pvals = de.stats.two_coef_z_test(theta_mle0=theta_mle0, theta_mle1=theta_mle1, theta_sd0=theta_sds,
                                         theta_sd1=theta_sds)
        
        # Compare p-value distribution under null model against uniform distribution.
        pval_h0 = stats.kstest(pvals, 'uniform').pvalue

        logging.getLogger("diffxpy").info('KS-test pvalue for null model match of z_test(): %f' % pval_h0)
        assert pval_h0 > 0.05, "KS-Test failed: pval_h0 is <= 0.05!"

        return True
    
    def test_wilcoxon(self, n: int = 1000, n_test: int = 100):
        """
        Test if de.stats.wilcoxon() generates a uniform p-value distribution
        if it is given data sampled from the null model. Returns the p-value
        of the two-side Kolmgorov-Smirnov test for equality of the observed 
        p-value distriubution and a uniform distribution.

        :param n: Number of tests to run.
        :param n_test: Sample size of each group in each test.
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)
        
        # Draw sample distribution parameters for each test:
        locs = np.random.normal(loc=0, scale=1, size=n)
        scales = np.exp(np.random.normal(loc=0, scale=0.5, size=n))
        
        # Draw two sets of samples  estimates for each test:
        x0 = np.vstack([np.random.normal(loc=locs[i], scale=scales[i], size=n_test) for i in range(n)]).T
        x1 = np.vstack([np.random.normal(loc=locs[i], scale=scales[i], size=n_test) for i in range(n)]).T
        
        # Compute p-value distribution under null model.
        pvals = de.stats.mann_whitney_u_test(x0=x0, x1=x1)
        
        # Compare p-value distribution under null model against uniform distribution.
        pval_h0 = stats.kstest(pvals, 'uniform').pvalue

        logging.getLogger("diffxpy").info('KS-test pvalue for null model match of wilcoxon(): %f' % pval_h0)
        assert pval_h0 > 0.05, "KS-Test failed: pval_h0 is <= 0.05!"

        return True
    
    def test_t_test_raw(self, n: int = 1000, n_test: int = 100):
        """
        Test if de.stats.t_test_raw() generates a uniform p-value distribution
        if it is given data sampled from the null model. Returns the p-value
        of the two-side Kolmgorov-Smirnov test for equality of the observed 
        p-value distriubution and a uniform distribution.

        :param n: int
            Number of tests to run.
        :param n_test: int
            Sample size of each group in each test.
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)
        
        # Draw sample distribution parameters for each test:
        locs = np.random.normal(loc=0, scale=1, size=n)
        scales = np.exp(np.random.normal(loc=0, scale=0.5, size=n))
        
        # Draw two sets of samples  estimates for each test:
        x0 = np.vstack([np.random.normal(loc=locs[i], scale=scales[i], size=n_test) for i in range(n)]).T
        x1 = np.vstack([np.random.normal(loc=locs[i], scale=scales[i], size=n_test) for i in range(n)]).T
        
        # Compute p-value distribution under null model.
        pvals = de.stats.t_test_raw(x0=x0, x1=x1)
        
        # Compare p-value distribution under null model against uniform distribution.
        pval_h0 = stats.kstest(pvals, 'uniform').pvalue

        logging.getLogger("diffxpy").info('KS-test pvalue for null model match of t_test_raw(): %f' % pval_h0)
        assert pval_h0 > 0.05, "KS-Test failed: pval_h0 is <= 0.05!"

        return True


if __name__ == '__main__':
    unittest.main()
