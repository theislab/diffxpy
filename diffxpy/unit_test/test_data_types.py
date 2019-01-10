import unittest
import logging

import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.sparse
import anndata

from batchglm.api.models.glm_nb import Simulator
import diffxpy.api as de


class TestDataTypes(unittest.TestCase):

    def test_sparse_anndata(self, n_cells: int = 2000, n_genes: int = 100):
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
            "condition": np.random.randint(2, size=sim.num_observations)
        })

        adata = anndata.AnnData(scipy.sparse.csr_matrix(sim.X.values))
        # X = adata.X
        test = de.test.wald(
            data=adata,
            factor_loc_totest="condition",
            formula="~ 1 + condition",
            sample_description=random_sample_description,
            quick_scale=True,
            training_strategy="DEFAULT",
            dtype="float64"
        )
        summary = test.summary()

        # Compare p-value distribution under null model against uniform distribution.
        pval_h0 = stats.kstest(test.pval, 'uniform').pvalue

        logging.getLogger("diffxpy").info('KS-test pvalue for null model match of wald(): %f' % pval_h0)
        assert pval_h0 > 0.05, "KS-Test failed: pval_h0 is <= 0.05!"

        return True


if __name__ == '__main__':
    unittest.main()
