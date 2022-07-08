import logging
import unittest
import numpy as np
import pandas as pd
import scipy.stats as stats
from batchglm.models.glm_nb import Model as NBModel

import diffxpy.api as de

class TestVsRest(unittest.TestCase):
    # NOTE: This test fails sometimes, and passes other times when the groups or loc are less extreme.
    def test_null_distribution_wald(self, n_cells: int = 2000, n_genes: int = 100, n_groups: int = 4):
        """
        Test if de.test_wald_loc() generates a uniform p-value distribution
        if it is given data simulated based on the null model. Returns the p-value
        of the two-side Kolmgorov-Smirnov test for equality of the observed
        p-value distriubution and a uniform distribution.

        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)
        model = NBModel()
        rand_fn_loc = lambda shape: np.random.uniform(9, 10, shape)
        rand_fn_scale = lambda shape: np.random.uniform(1, 2, shape)
        model.generate_artificial_data(
            n_obs=n_cells,
            n_vars=n_genes,
            num_batches=0,
            num_conditions=0,
            rand_fn_loc=rand_fn_loc,
            rand_fn_scale=rand_fn_scale
        )

        random_sample_description = pd.DataFrame({
            "condition": np.random.randint(n_groups, size=n_cells)
        })

        test = de.test.versus_rest(
            data=model.x,
            gene_names=model.features,
            grouping="condition",
            test="wald",
            noise_model="nb",
            sample_description=random_sample_description,
            # batch_size=(500, 500), # why was this here?
            training_strategy="DEFAULT",
            dtype="float64"
        )
        summary = test.summary()

        # Compare p-value distribution under null model against uniform distribution.
        pval_h0 = stats.kstest(test.pval.flatten(), 'uniform').pvalue

        logging.getLogger("diffxpy").info('KS-test pvalue for null model match of test_wald_loc(): %f' % pval_h0)
        assert pval_h0 > 0.05, "KS-Test failed: pval_h0=%f is <= 0.05!" % np.round(pval_h0, 5)

        return True

    def test_null_distribution_lrt(self, n_cells: int = 2000, n_genes: int = 100):
        """
        Test if de.test_wald_loc() generates a uniform p-value distribution
        if it is given data simulated based on the null model. Returns the p-value
        of the two-side Kolmgorov-Smirnov test for equality of the observed
        p-value distriubution and a uniform distribution.

        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.ERROR)
        model = NBModel()
        model.generate_artificial_data(
            n_obs=n_cells,
            n_vars=n_genes,
            num_batches=0,
            num_conditions=0
        )

        random_sample_description = pd.DataFrame({
            "condition": np.random.randint(2, size=n_cells)
        })

        test = de.test.versus_rest(
            data=model.x,
            gene_names=model.features,
            grouping="condition",
            test="lrt",
            noise_model="nb",
            sample_description=random_sample_description,
            # batch_size=(500, 500), # why was this here?
            training_strategy="DEFAULT",
            dtype="float64"
        )
        summary = test.summary()

        # Compare p-value distribution under null model against uniform distribution.
        pval_h0 = stats.kstest(test.pval.flatten(), 'uniform').pvalue

        logging.getLogger("diffxpy").info('KS-test pvalue for null model match of test_lrt(): %f' % pval_h0)
        assert pval_h0 > 0.05, "KS-Test failed: pval_h0=%f is <= 0.05!" % np.round(pval_h0, 5)

        return True

    def test_null_distribution_rank(self, n_cells: int = 2000, n_genes: int = 100, n_groups: int = 2):
        """
        Test if de.test_wald_loc() generates a uniform p-value distribution
        if it is given data simulated based on the null model. Returns the p-value
        of the two-side Kolmgorov-Smirnov test for equality of the observed
        p-value distriubution and a uniform distribution.

        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)
        model = NBModel()
        model.generate_artificial_data(
            n_obs=n_cells,
            n_vars=n_genes,
            num_batches=0,
            num_conditions=0
        )

        random_sample_description = pd.DataFrame({
            "condition": np.random.randint(n_groups, size=n_cells)
        })

        test = de.test.versus_rest(
            data=model.x,
            gene_names=model.features,
            grouping="condition",
            test="rank",
            sample_description=random_sample_description,
            dtype="float64"
        )
        summary = test.summary()

        # Compare p-value distribution under null model against uniform distribution.
        pval_h0 = stats.kstest(test.pval.flatten(), 'uniform').pvalue

        logging.getLogger("diffxpy").info('KS-test pvalue for null model match of test_wald_loc(): %f' % pval_h0)
        assert pval_h0 > 0.05, "KS-Test failed: pval_h0=%f is <= 0.05!" % np.round(pval_h0, 5)

        return True

    def test_null_distribution_ttest(self, n_cells: int = 2000, n_genes: int = 100, n_groups: int = 2):
        """
        Test if de.test_wald_loc() generates a uniform p-value distribution
        if it is given data simulated based on the null model. Returns the p-value
        of the two-side Kolmgorov-Smirnov test for equality of the observed
        p-value distriubution and a uniform distribution.

        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)
        model = NBModel()
        model.generate_artificial_data(
            n_obs=n_cells,
            n_vars=n_genes,
            num_batches=0,
            num_conditions=0
        )

        random_sample_description = pd.DataFrame({
            "condition": np.random.randint(n_groups, size=n_cells)
        })

        test = de.test.versus_rest(
            data=model.x,
            gene_names=model.features,
            grouping="condition",
            test="t-test",
            sample_description=random_sample_description,
            dtype="float64"
        )
        summary = test.summary()

        # Compare p-value distribution under null model against uniform distribution.
        pval_h0 = stats.kstest(test.pval.flatten(), 'uniform').pvalue

        logging.getLogger("diffxpy").info('KS-test pvalue for null model match of test_wald_loc(): %f' % pval_h0)
        assert pval_h0 > 0.05, "KS-Test failed: pval_h0=%f is <= 0.05!" % np.round(pval_h0, 5)

        return True


if __name__ == '__main__':
    unittest.main()
