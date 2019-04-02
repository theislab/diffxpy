import unittest
import logging
import numpy as np
import pandas as pd
import scipy.stats as stats

from batchglm.api.models.glm_nb import Simulator
import diffxpy.api as de


class TestSingleNull(unittest.TestCase):

    def test_null_distribution_wald(self, n_cells: int = 2000, n_genes: int = 200):
        """
        Test if de.wald() generates a uniform p-value distribution
        if it is given data simulated based on the null model. Returns the p-value
        of the two-side Kolmgorov-Smirnov test for equality of the observed 
        p-value distribution and a uniform distribution.

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
            "condition": np.random.randint(2, size=sim.num_observations),
            "batch": np.random.randint(2, size=sim.num_observations)
        })

        test = de.test.wald(
            data=sim.X,
            factor_loc_totest="condition",
            formula_loc="~ 1 + condition + batch",
            sample_description=random_sample_description,
            batch_size=500,
            training_strategy="DEFAULT",
            dtype="float64"
        )
        summary = test.summary()

        # Compare p-value distribution under null model against uniform distribution.
        pval_h0 = stats.kstest(test.pval, 'uniform').pvalue

        logging.getLogger("diffxpy").info('KS-test pvalue for null model match of wald(): %f' % pval_h0)
        assert pval_h0 > 0.05, "KS-Test failed: pval_h0=%f is <= 0.05!" % np.round(pval_h0, 5)

        return True

    def test_null_distribution_wald_multi(self, n_cells: int = 2000, n_genes: int = 200):
        """
        Test if de.wald() (multivariate mode) generates a uniform p-value distribution
        if it is given data simulated based on the null model. Returns the p-value
        of the two-side Kolmgorov-Smirnov test for equality of the observed
        p-value distribution and a uniform distribution.

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

        test = de.test.wald(
            data=sim.X,
            factor_loc_totest="condition",
            formula_loc="~ 1 + condition",
            sample_description=random_sample_description,
            training_strategy="DEFAULT",
            dtype="float64"
        )
        summary = test.summary()

        # Compare p-value distribution under null model against uniform distribution.
        pval_h0 = stats.kstest(test.pval, 'uniform').pvalue

        logging.getLogger("diffxpy").info('KS-test pvalue for null model match of wald(): %f' % pval_h0)
        assert pval_h0 > 0.05, "KS-Test failed: pval_h0=%f is <= 0.05!" % np.round(pval_h0, 5)

        return True

    def test_null_distribution_lrt(self, n_cells: int = 2000, n_genes: int = 200):
        """
        Test if de.lrt() generates a uniform p-value distribution
        if it is given data simulated based on the null model. Returns the p-value
        of the two-side Kolmgorov-Smirnov test for equality of the observed
        p-value distribution and a uniform distribution.

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
            "condition": np.random.randint(2, size=sim.num_observations)
        })

        test = de.test.lrt(
            data=sim.X,
            full_formula_loc="~ 1 + condition",
            full_formula_scale="~ 1",
            reduced_formula_loc="~ 1",
            reduced_formula_scale="~ 1",
            sample_description=random_sample_description,
            training_strategy="DEFAULT",
            dtype="float64"
        )
        summary = test.summary()

        # Compare p-value distribution under null model against uniform distribution.
        pval_h0 = stats.kstest(test.pval, 'uniform').pvalue

        logging.getLogger("diffxpy").info('KS-test pvalue for null model match of lrt(): %f' % pval_h0)
        assert pval_h0 > 0.05, "KS-Test failed: pval_h0=%f is <= 0.05!" % np.round(pval_h0, 5)

        return True

    def test_null_distribution_ttest(self, n_cells: int = 2000, n_genes: int = 200):
        """
        Test if de.t_test() generates a uniform p-value distribution
        if it is given data simulated based on the null model. Returns the p-value
        of the two-side Kolmgorov-Smirnov test for equality of the observed
        p-value distribution and a uniform distribution.

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
            "condition": np.random.randint(2, size=sim.num_observations)
        })

        test = de.test.t_test(
            data=sim.X,
            grouping="condition",
            sample_description=random_sample_description,
            is_logged=False,
            dtype="float64"
        )
        summary = test.summary()

        # Compare p-value distribution under null model against uniform distribution.
        pval_h0 = stats.kstest(test.pval, 'uniform').pvalue

        logging.getLogger("diffxpy").info('KS-test pvalue for null model match of t_test(): %f' % pval_h0)
        assert pval_h0 > 0.05, "KS-Test failed: pval_h0=%f is <= 0.05!" % np.round(pval_h0, 5)

        return True

    def test_null_distribution_wilcoxon(self, n_cells: int = 2000, n_genes: int = 200):
        """
        Test if de.wilcoxon() generates a uniform p-value distribution
        if it is given data simulated based on the null model. Returns the p-value
        of the two-side Kolmgorov-Smirnov test for equality of the observed
        p-value distribution and a uniform distribution.

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
            "condition": np.random.randint(2, size=sim.num_observations)
        })

        test = de.test.rank_test(
            data=sim.X,
            grouping="condition",
            sample_description=random_sample_description,
            dtype="float64"
        )
        summary = test.summary()

        # Compare p-value distribution under null model against uniform distribution.
        pval_h0 = stats.kstest(test.pval, 'uniform').pvalue

        logging.getLogger("diffxpy").info('KS-test pvalue for null model match of wilcoxon(): %f' % pval_h0)
        assert pval_h0 > 0.05, "KS-Test failed: pval_h0=%f is <= 0.05!" % np.round(pval_h0, 5)

        return True
    

if __name__ == '__main__':
    unittest.main()
