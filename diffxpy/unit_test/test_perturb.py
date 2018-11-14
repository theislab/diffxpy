import unittest

import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.sparse
import anndata

from batchglm.api.models.nb_glm import Simulator, Estimator, InputData
import diffxpy.api as de


class TestSingle(unittest.TestCase):

    def test_null_distribution_z(self, n_cells: int = 200, n_genes: int = 20):
        """
        Test if de.perturb() generates a uniform p-value distribution
        if it is given data simulated based on the null model. Returns the p-value
        of the two-side Kolmgorov-Smirnov test for equality of the observed 
        p-value distriubution and a uniform distribution.

        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        """

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

        print('KS-test pvalue for null model match of wald(): %f' % pval_h0)

        assert pval_h0 > 0.05, "KS-Test failed: pval_h0 is <= 0.05!"

        return pval_h0


if __name__ == '__main__':
    unittest.main()
