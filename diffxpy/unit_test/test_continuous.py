import unittest

import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.sparse
import anndata
import logging

from batchglm.api.models.nb_glm import Simulator, Estimator, InputData
import diffxpy.api as de


class TestContinuous(unittest.TestCase):

    def test_forfatal_functions(self):
        """
        Test if de.test.continuous() DifferentialExpressionTestSingle object functions work fine.

        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        """
        logging.getLogger('diffxpy').addFilter('DEBUG')

        num_observations = 10
        num_features = 2

        sim = Simulator(num_observations=num_observations, num_features=num_features)
        sim.generate_sample_description(num_batches=0, num_conditions=0)
        sim.generate()

        random_sample_description = pd.DataFrame({
            "pseudotime": np.random.random(size=sim.num_observations),
            "batch": np.random.randint(2, size=sim.num_observations)
        })

        test = de.test.continuous_1d(
            data=sim.X,
            continuous="pseudotime",
            df=3,
            formula_loc="~ 1 + pseudotime + batch",
            formula_scale="~ 1",
            factor_loc_totest="pseudotime",
            test="wald",
            sample_description=random_sample_description,
            quick_scale=True,
            batch_size=None,
            dtype="float64"
        )

        summary = test.summary()
        ids = test.gene_ids

        # 1. Test all additional functions which depend on model computation:
        # 1.1. Only continuous model:
        temp = test.log_fold_change(genes=ids, nonnumeric=False)
        temp = test.max(genes=ids, nonnumeric=False)
        temp = test.min(genes=ids, nonnumeric=False)
        temp = test.argmax(genes=ids, nonnumeric=False)
        temp = test.argmin(genes=ids, nonnumeric=False)
        temp = test.summary(nonnumeric=False)
        # 1.2. Full model:
        temp = test.log_fold_change(genes=ids, nonnumeric=True)
        temp = test.max(genes=ids, nonnumeric=True)
        temp = test.min(genes=ids, nonnumeric=True)
        temp = test.argmax(genes=ids, nonnumeric=True)
        temp = test.argmin(genes=ids, nonnumeric=True)
        temp = test.summary(nonnumeric=True)


    def test_null_distribution_wald(self, n_cells: int = 10, n_genes: int = 2):
        """
        Test if de.test.continuous() generates a uniform p-value distribution in the wald test
        if it is given data simulated based on the null model. Returns the p-value
        of the two-side Kolmgorov-Smirnov test for equality of the observed 
        p-value distriubution and a uniform distribution.

        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        """
        logging.getLogger('diffxpy').addFilter('DEBUG')

        sim = Simulator(num_observations=n_cells, num_features=n_genes)
        sim.generate_sample_description(num_batches=0, num_conditions=0)
        sim.generate()

        random_sample_description = pd.DataFrame({
            "pseudotime": np.random.random(size=sim.num_observations)
        })

        test = de.test.continuous_1d(
            data=sim.X,
            continuous="pseudotime",
            df=3,
            formula_loc="~ 1 + pseudotime",
            formula_scale="~ 1",
            factor_loc_totest="pseudotime",
            test="wald",
            sample_description=random_sample_description,
            quick_scale=True,
            batch_size=None,
            dtype="float64"
        )
        summary = test.summary()

        # Compare p-value distribution under null model against uniform distribution.
        pval_h0 = stats.kstest(test.pval, 'uniform').pvalue

        print('KS-test pvalue for null model match of wald(): %f' % pval_h0)
        print(test.pval)

        assert pval_h0 > 0.05, "KS-Test failed: pval_h0 is <= 0.05!"

        return pval_h0

    def test_null_distribution_lrt(self, n_cells: int = 10, n_genes: int = 2):
        """
        Test if de.test.continuous() generates a uniform p-value distribution in lrt
        if it is given data simulated based on the null model. Returns the p-value
        of the two-side Kolmgorov-Smirnov test for equality of the observed
        p-value distriubution and a uniform distribution.

        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        """
        logging.getLogger('diffxpy').addFilter('DEBUG')

        sim = Simulator(num_observations=n_cells, num_features=n_genes)
        sim.generate_sample_description(num_batches=0, num_conditions=0)
        sim.generate()

        random_sample_description = pd.DataFrame({
            "pseudotime": np.random.random(size=sim.num_observations)
        })

        test = de.test.continuous_1d(
            data=sim.X,
            continuous="pseudotime",
            df=3,
            formula_loc="~ 1 + pseudotime",
            formula_scale="~ 1",
            factor_loc_totest="pseudotime",
            test="lrt",
            sample_description=random_sample_description,
            quick_scale=True,
            batch_size=None,
            dtype="float64"
        )
        summary = test.summary()

        # Compare p-value distribution under null model against uniform distribution.
        pval_h0 = stats.kstest(test.pval, 'uniform').pvalue

        print('KS-test pvalue for null model match of wald(): %f' % pval_h0)
        print(test.pval)

        assert pval_h0 > 0.05, "KS-Test failed: pval_h0 is <= 0.05!"

        return pval_h0

if __name__ == '__main__':
    unittest.main()
