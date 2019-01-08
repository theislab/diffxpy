import unittest

import numpy as np
import pandas as pd
import scipy.stats as stats
import logging

import batchglm.api as glm
from batchglm.api.models.glm_nb import Simulator, Estimator, InputData
import diffxpy.api as de

glm.setup_logging(verbosity="INFO", stream="STDOUT")
logging.getLogger("tensorflow").setLevel(logging.INFO)


class TestSingleFeaturewiseTermination(unittest.TestCase):

    def test_null_distribution_wald_by_gene_nr(self, n_cells: int = 2000, n_genes: int = 50):
        """
        Test if de.wald() generates a uniform p-value distribution
        if it is given data simulated based on the null model. Returns the p-value
        of the two-side Kolmgorov-Smirnov test for equality of the observed 
        p-value distribution and a uniform distribution.

        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        """

        sim = Simulator(num_observations=n_cells, num_features=n_genes)
        sim.generate_sample_description(num_batches=0, num_conditions=0)
        sim.generate()

        random_sample_description = pd.DataFrame({
            "condition": np.random.randint(2, size=sim.num_observations)
        })

        test = de.test.wald(
            data=sim.X,
            factor_loc_totest="condition",
            formula="~ 1 + condition",
            init_a="standard",
            init_b="standard",
            sample_description=random_sample_description,
            training_strategy="BY_GENE_NR",
            dtype="float64"
        )
        summary = test.summary()

        # Compare p-value distribution under null model against uniform distribution.
        pval_h0 = stats.kstest(test.pval, 'uniform').pvalue

        print('KS-test pvalue for null model match of wald() with stratgey BY_GENE_NR: %f' % pval_h0)

        assert pval_h0 > 0.05, "KS-Test failed: pval_h0 is <= 0.05!"

        return pval_h0

    def test_null_distribution_wald_by_gene_adam(self, n_cells: int = 2000, n_genes: int = 50):
        """
        Test if de.wald() generates a uniform p-value distribution
        if it is given data simulated based on the null model. Returns the p-value
        of the two-side Kolmgorov-Smirnov test for equality of the observed
        p-value distribution and a uniform distribution.

        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        """

        sim = Simulator(num_observations=n_cells, num_features=n_genes)
        sim.generate_sample_description(num_batches=0, num_conditions=0)
        sim.generate()

        random_sample_description = pd.DataFrame({
            "condition": np.random.randint(2, size=sim.num_observations)
        })

        test = de.test.wald(
            data=sim.X,
            factor_loc_totest="condition",
            formula="~ 1 + condition",
            init_a="standard",
            init_b="standard",
            sample_description=random_sample_description,
            training_strategy="BY_GENE_ADAM",
            dtype="float64"
        )
        summary = test.summary()

        # Compare p-value distribution under null model against uniform distribution.
        pval_h0 = stats.kstest(test.pval, 'uniform').pvalue

        print('KS-test pvalue for null model match of wald() with stratgey BY_GENE_NR: %f' % pval_h0)

        assert pval_h0 > 0.05, "KS-Test failed: pval_h0 is <= 0.05!"

        return pval_h0


if __name__ == '__main__':
    unittest.main()
