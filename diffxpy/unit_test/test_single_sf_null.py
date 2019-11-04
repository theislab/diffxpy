import unittest
import logging
import numpy as np
import pandas as pd
import scipy.stats as stats

import diffxpy.api as de


class _TestSingleSfNull:

    def _test_null_distribution_wald(
            self,
            n_cells: int,
            n_genes: int,
            noise_model: str
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
            from batchglm.api.models.numpy.glm_nb import Simulator
            rand_fn_scale = lambda shape: np.random.uniform(1, 2, shape)
        elif noise_model == "norm":
            from batchglm.api.models.numpy.glm_norm import Simulator
            rand_fn_scale = lambda shape: np.random.uniform(1, 2, shape)
        else:
            raise ValueError("noise model %s not recognized" % noise_model)

        sim = Simulator(num_observations=n_cells, num_features=n_genes)
        sim.generate_sample_description(num_batches=0, num_conditions=0)
        sim.generate_params(rand_fn_scale=rand_fn_scale)
        sim.generate_data()

        random_sample_description = pd.DataFrame({
            "condition": np.random.randint(2, size=sim.nobs),
            "batch": np.random.randint(2, size=sim.nobs)
        })
        random_sf = np.random.uniform(0.5, 1.5, sim.nobs)

        test = de.test.wald(
            data=sim.input_data,
            sample_description=random_sample_description,
            factor_loc_totest="condition",
            formula_loc="~ 1 + condition + batch",
            size_factors=random_sf,
            batch_size=500,
            noise_model=noise_model,
            training_strategy="DEFAULT",
            dtype="float64"
        )
        _ = test.summary()

        # Compare p-value distribution under null model against uniform distribution.
        pval_h0 = stats.kstest(test.pval, 'uniform').pvalue

        logging.getLogger("diffxpy").info('KS-test pvalue for null model match of wald(): %f' % pval_h0)
        assert pval_h0 > 0.05, ("KS-Test failed: pval_h0=%f is <= 0.05!" % np.round(pval_h0, 5))

        return True


class TestSingleSfNullNb(_TestSingleSfNull, unittest.TestCase):
    """
    Negative binomial noise model unit tests that test whether a test generates uniformly
    distributed p-values if data are sampled from the null model.
    """

    def test_null_distribution_wald_nb(
            self,
            n_cells: int = 2000,
            n_genes: int = 200
    ):
        """
        Test if wald() generates a uniform p-value distribution for "nb" noise model.

        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)

        np.random.seed(1)
        return self._test_null_distribution_wald(
            n_cells=n_cells,
            n_genes=n_genes,
            noise_model="nb"
        )


class TestSingleSfNullNorm(_TestSingleSfNull, unittest.TestCase):
    """
    Normal noise model unit tests that test whether a test generates uniformly
    distributed p-values if data are sampled from the null model.
    """
    def test_null_distribution_wald_norm(
            self,
            n_cells: int = 200,
            n_genes: int = 200
    ):
        """
        Test if wald() generates a uniform p-value distribution for "norm" noise model.

        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)

        np.random.seed(1)
        return self._test_null_distribution_wald(
            n_cells=n_cells,
            n_genes=n_genes,
            noise_model="norm"
        )


if __name__ == '__main__':
    unittest.main()
