import unittest
import logging
import numpy as np
import pandas as pd
import scipy.stats as stats
from batchglm.models.glm_nb import Model as NBModel
from batchglm.models.glm_norm import Model as NormModel

import diffxpy.api as de


class _TestSingleNullBackends:

    def _test_null_distribution_wald(
            self,
            n_cells: int,
            n_genes: int,
            noise_model: str,
            backend: str
    ):
        """
        Test if de.wald() generates a uniform p-value distribution
        if it is given data simulated based on the null model. Returns the p-value
        of the two-side Kolmgorov-Smirnov test for equality of the observed 
        p-value distribution and a uniform distribution.

        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        :param noise_model: Noise model to use for data fitting.
        """
        if noise_model == "nb":
            model = NBModel()
            rand_fn_scale = lambda shape: np.random.uniform(1, 2, shape)
        elif noise_model == "norm":
            model = NormModel()
            rand_fn_scale = lambda shape: np.random.uniform(1, 2, shape)


        model.generate_artificial_data(
            n_obs=n_cells,
            n_vars=n_genes,
            num_batches=0,
            num_conditions=0,
            rand_fn_scale=rand_fn_scale
        )

        random_sample_description = pd.DataFrame({
            "condition": np.random.randint(2, size=n_cells),
            "batch": np.random.randint(2, size=n_cells)
        })

        test = de.test.wald(
            data=model.x,
            gene_names=model.features,
            sample_description=random_sample_description,
            factor_loc_totest="condition",
            formula_loc="~ 1 + condition + batch",
            noise_model=noise_model,
            backend=backend
        )
        _ = test.summary()

        # Compare p-value distribution under null model against uniform distribution.
        pval_h0 = stats.kstest(test.pval, 'uniform').pvalue

        logging.getLogger("diffxpy").info('KS-test pvalue for null model match of wald(): %f' % pval_h0)
        assert pval_h0 > 0.05, ("KS-Test failed: pval_h0=%f is <= 0.05!" % np.round(pval_h0, 5))

        return True


class TestSingleNullBackendsNb(_TestSingleNullBackends, unittest.TestCase):
    """
    Negative binomial noise model unit tests that test whether a test generates uniformly
    distributed p-values if data are sampled from the null model.
    """

    def test_null_distribution_wald_nb_numpy(
            self,
            n_cells: int = 2000,
            n_genes: int = 200
    ):
        """
        Test if wald() generates a uniform p-value distribution for "nb" noise model under numpy backend

        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)

        np.random.seed(1)
        _ = self._test_null_distribution_wald(
            n_cells=n_cells,
            n_genes=n_genes,
            noise_model="nb",
            backend="numpy"
        )


if __name__ == '__main__':
    unittest.main()
