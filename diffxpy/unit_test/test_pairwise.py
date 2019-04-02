import logging
import unittest
import numpy as np
import pandas as pd
import scipy.stats as stats

from batchglm.api.models.glm_nb import Simulator
import diffxpy.api as de


class TestPairwiseNull(unittest.TestCase):

    def test_null_distribution_ztest(self, n_cells: int = 2000, n_genes: int = 100, n_groups=2):
        """
        Test if de.wald() generates a uniform p-value distribution
        if it is given data simulated based on the null model. Returns the p-value
        of the two-side Kolmgorov-Smirnov test for equality of the observed 
        p-value distriubution and a uniform distribution.

        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)

        sim = Simulator(num_observations=n_cells, num_features=n_genes)
        sim.generate_sample_description(num_batches=0, num_conditions=0)
        sim.generate()

        random_sample_description = pd.DataFrame({
            "condition": np.random.randint(n_groups, size=sim.num_observations)
        })

        test = de.test.pairwise(
            data=sim.X,
            grouping="condition",
            test="z-test",
            noise_model="nb",
            sample_description=random_sample_description,
            dtype="float64"
        )
        summary = test.summary()

        # Compare p-value distribution under null model against uniform distribution.
        pval_h0 = stats.kstest(test.pval[~np.eye(test.pval.shape[0]).astype(bool)].flatten(), 'uniform').pvalue

        logging.getLogger("diffxpy").info('KS-test pvalue for null model match of wald(): %f' % pval_h0)
        assert pval_h0 > 0.05, "KS-Test failed: pval_h0=%f is <= 0.05!" % np.round(pval_h0, 5)

        return True

    def test_null_distribution_z_lazy(self, n_cells: int = 2000, n_genes: int = 100):
        """
        Test if de.pairwise() generates a uniform p-value distribution for lazy z-tests
        if it is given data simulated based on the null model. Returns the p-value
        of the two-side Kolmgorov-Smirnov test for equality of the observed
        p-value distriubution and a uniform distribution.

        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)

        sim = Simulator(num_observations=n_cells, num_features=n_genes)
        sim.generate_sample_description(num_batches=0, num_conditions=0)
        sim.generate()

        random_sample_description = pd.DataFrame({
            "condition": np.random.randint(4, size=sim.num_observations)
        })

        test = de.test.pairwise(
            data=sim.X,
            grouping="condition",
            test='z-test',
            lazy=True,
            noise_model="nb",
            pval_correction="global",
            quick_scale=True,
            sample_description=random_sample_description,
            dtype="float64"
        )

        # Compare p-value distribution under null model against uniform distribution.
        pvals = test.pval_pairs(groups0=0, groups1=1)
        pval_h0 = stats.kstest(pvals.flatten(), 'uniform').pvalue

        logging.getLogger("diffxpy").info('KS-test pvalue for null model match of wald(): %f' % pval_h0)
        assert pval_h0 > 0.05, "KS-Test failed: pval_h0=%f is <= 0.05!" % np.round(pval_h0, 5)

        return True

    def test_null_distribution_lrt(self, n_cells: int = 2000, n_genes: int = 100, n_groups=2):
        """
        Test if de.wald() generates a uniform p-value distribution
        if it is given data simulated based on the null model. Returns the p-value
        of the two-side Kolmgorov-Smirnov test for equality of the observed
        p-value distriubution and a uniform distribution.

        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)

        sim = Simulator(num_observations=n_cells, num_features=n_genes)
        sim.generate_sample_description(num_batches=0, num_conditions=0)
        sim.generate()

        random_sample_description = pd.DataFrame({
            "condition": np.random.randint(n_groups, size=sim.num_observations)
        })

        test = de.test.pairwise(
            data=sim.X,
            grouping="condition",
            test="lrt",
            noise_model="nb",
            sample_description=random_sample_description,
            dtype="float64"
        )

        # Compare p-value distribution under null model against uniform distribution.
        pval_h0 = stats.kstest(test.pval[~np.eye(test.pval.shape[0]).astype(bool)].flatten(), 'uniform').pvalue

        logging.getLogger("diffxpy").info('KS-test pvalue for null model match of wald(): %f' % pval_h0)
        assert pval_h0 > 0.05, "KS-Test failed: pval_h0=%f is <= 0.05!" % np.round(pval_h0, 5)

        return True

    def test_null_distribution_ttest(self, n_cells: int = 2000, n_genes: int = 10000, n_groups=2):
        """
        Test if de.wald() generates a uniform p-value distribution
        if it is given data simulated based on the null model. Returns the p-value
        of the two-side Kolmgorov-Smirnov test for equality of the observed
        p-value distriubution and a uniform distribution.

        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)

        sim = Simulator(num_observations=n_cells, num_features=n_genes)
        sim.generate_sample_description(num_batches=0, num_conditions=0)
        sim.generate()

        random_sample_description = pd.DataFrame({
            "condition": np.random.randint(n_groups, size=sim.num_observations)
        })

        test = de.test.pairwise(
            data=sim.X,
            grouping="condition",
            test="t-test",
            sample_description=random_sample_description,
        )
        summary = test.summary()

        # Compare p-value distribution under null model against uniform distribution.
        pval_h0 = stats.kstest(test.pval[~np.eye(test.pval.shape[0]).astype(bool)].flatten(), 'uniform').pvalue

        logging.getLogger("diffxpy").info('KS-test pvalue for null model match of wald(): %f' % pval_h0)
        assert pval_h0 > 0.05, "KS-Test failed: pval_h0=%f is <= 0.05!" % np.round(pval_h0, 5)

        return True

    def test_null_distribution_wilcoxon(self, n_cells: int = 2000, n_genes: int = 10000, n_groups=2):
        """
        Test if de.wald() generates a uniform p-value distribution
        if it is given data simulated based on the null model. Returns the p-value
        of the two-side Kolmgorov-Smirnov test for equality of the observed
        p-value distriubution and a uniform distribution.

        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)

        sim = Simulator(num_observations=n_cells, num_features=n_genes)
        sim.generate_sample_description(num_batches=0, num_conditions=0)
        sim.generate()

        random_sample_description = pd.DataFrame({
            "condition": np.random.randint(n_groups, size=sim.num_observations)
        })

        test = de.test.pairwise(
            data=sim.X,
            grouping="condition",
            test="wilcoxon",
            sample_description=random_sample_description,
        )
        summary = test.summary()

        # Compare p-value distribution under null model against uniform distribution.
        pval_h0 = stats.kstest(test.pval[~np.eye(test.pval.shape[0]).astype(bool)].flatten(), 'uniform').pvalue

        logging.getLogger("diffxpy").info('KS-test pvalue for null model match of wald(): %f' % pval_h0)
        assert pval_h0 > 0.05, "KS-Test failed: pval_h0=%f is <= 0.05!" % np.round(pval_h0, 5)

        return True


class TestPairwiseDE(unittest.TestCase):

    def test_ztest_de(self, n_cells: int = 2000, n_genes: int = 500):
        """
        Test if de.lrt() generates a uniform p-value distribution
        if it is given data simulated based on the null model. Returns the p-value
        of the two-side Kolmgorov-Smirnov test for equality of the observed
        p-value distriubution and a uniform distribution.

        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)

        num_non_de = n_genes // 2
        sim = Simulator(num_observations=n_cells, num_features=n_genes)
        sim.generate_sample_description(num_batches=0, num_conditions=2)
        # simulate: coefficients ~ log(N(1, 0.5)).
        # re-sample if N(1, 0.5) <= 0
        sim.generate_params(rand_fn=lambda shape: 1 + stats.truncnorm.rvs(-1 / 0.5, np.infty, scale=0.5, size=shape))
        sim.params["a"][1, :num_non_de] = 0
        sim.params["b"][1, :num_non_de] = 0
        sim.params["isDE"] = ("features",), np.arange(n_genes) >= num_non_de
        sim.generate_data()

        sample_description = sim.sample_description

        test = de.test.pairwise(
            data=sim.X,
            grouping="condition",
            test="z-test",
            noise_model="nb",
            sample_description=sample_description,
        )
        summary = test.summary()

        frac_nonde_sig = np.mean(
            np.sum(test.qval[~np.eye(test.pval.shape[0]).astype(bool), :num_non_de] < 0.05) /
            (2 * num_non_de)
        )
        frac_de_sig = np.mean(
            np.sum(test.qval[~np.eye(test.pval.shape[0]).astype(bool), num_non_de:] < 0.05) /
            (2 * (n_genes - num_non_de))
        )
        logging.getLogger("diffxpy").info('fraction of non-DE genes with q-value < 0.05: %.1f%%' %
                                          str(np.round(100. * frac_nonde_sig, 3)))
        logging.getLogger("diffxpy").info('fraction of DE genes with q-value < 0.05: %.1f%%' %
                                          str(np.round(100. * frac_de_sig, 3)))

        assert frac_de_sig > 0.5, "too many DE"
        assert frac_nonde_sig < 0.5, "too many non-DE"
        return True


if __name__ == '__main__':
    unittest.main()
