import unittest
import logging
import numpy as np
import pandas as pd

import diffxpy.api as de
from batchglm.models.glm_nb import Model as NBModel
from batchglm.models.glm_norm import Model as NormModel

class _TestFit:

    def _test_model_fit(
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
            model = NBModel()
            rand_fn_scale = lambda shape: np.random.uniform(1, 2, shape)
        elif noise_model == "norm":
            model = NormModel()
            rand_fn_scale = lambda shape: np.random.uniform(1, 2, shape)
        else:
            raise ValueError("noise model %s not recognized" % noise_model)

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

        _ = de.fit.model(
            data=model.x,
            gene_names=model.features,
            sample_description=random_sample_description,
            formula_loc="~ 1 + condition + batch",
            noise_model=noise_model
        )
        return True

    def _test_model_fit_partition(
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
            model = NBModel()
            rand_fn_scale = lambda shape: np.random.uniform(1, 2, shape)
        elif noise_model == "norm":
            model = NormModel()
            rand_fn_scale = lambda shape: np.random.uniform(1, 2, shape)
        else:
            raise ValueError("noise model %s not recognized" % noise_model)

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

        partition = de.fit.partition(
            data=model.x,
            gene_names=model.features,
            sample_description=random_sample_description,
            parts="condition"
        )
        estim = partition.model(
            formula_loc="~ 1 + batch",
            noise_model=noise_model
        )
        return True

    def _test_residuals_fit(
            self,
            n_cells: int,
            n_genes: int,
            noise_model: str
    ):
        """
        Test if de.wald() (multivariate mode) generates a uniform p-value distribution
        if it is given data simulated based on the null model. Returns the p-value
        of the two-side Kolmgorov-Smirnov test for equality of the observed
        p-value distribution and a uniform distribution.

        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        :param noise_model: Noise model to use for data fitting.
        """
        if noise_model == "nb":
            model = NBModel()
        elif noise_model == "norm":
            model = NormModel()
        else:
            raise ValueError("noise model %s not recognized" % noise_model)

        model.generate_artificial_data(
            n_obs=n_cells,
            n_vars=n_genes,
            num_batches=0,
            num_conditions=0,
        )

        random_sample_description = pd.DataFrame({
            "condition": np.random.randint(2, size=n_cells),
            "batch": np.random.randint(2, size=n_cells)
        })

        res = de.fit.residuals(
            data=model.x,
            gene_names=model.features,
            sample_description=random_sample_description,
            formula_loc="~ 1 + condition + batch",
            noise_model=noise_model
        )
        return True


class TestFitNb(_TestFit, unittest.TestCase):
    """
    Negative binomial noise model unit tests that tests whether model fit relay works.
    """

    def test_model_fit(
            self,
            n_cells: int = 2000,
            n_genes: int = 2
    ):
        """
        Test if model for "nb" noise model works.

        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)

        np.random.seed(1)
        return self._test_model_fit(
            n_cells=n_cells,
            n_genes=n_genes,
            noise_model="nb"
        )

    def test_model_fit_partition(
            self,
            n_cells: int = 2000,
            n_genes: int = 2
    ):
        """
        Test if partitioned model for "nb" noise model works.

        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)

        np.random.seed(1)
        return self._test_model_fit_partition(
            n_cells=n_cells,
            n_genes=n_genes,
            noise_model="nb"
        )

    def test_residuals_fit(
            self,
            n_cells: int = 2000,
            n_genes: int = 2
    ):
        """
        Test if residual fit for "nb" noise model works.

        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)

        np.random.seed(1)
        return self._test_residuals_fit(
            n_cells=n_cells,
            n_genes=n_genes,
            noise_model="nb"
        )


class TestFitNorm(_TestFit, unittest.TestCase):
    """
    Normal noise model unit tests that tests whether model fit relay works.
    """

    def test_model_fit(
            self,
            n_cells: int = 2000,
            n_genes: int = 2
    ):
        """
        Test if model fit for "norm" noise model works.

        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)

        np.random.seed(1)
        return self._test_model_fit(
            n_cells=n_cells,
            n_genes=n_genes,
            noise_model="norm"
        )

    def test_model_fit_partition(
            self,
            n_cells: int = 2000,
            n_genes: int = 2
    ):
        """
        Test if partitioned model fit for "norm" noise model works.

        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)

        np.random.seed(1)
        return self._test_model_fit_partition(
            n_cells=n_cells,
            n_genes=n_genes,
            noise_model="norm"
        )

    def test_residuals_fit(
            self,
            n_cells: int = 2000,
            n_genes: int = 2
    ):
        """
        Test if residual fit for "norm" noise model works.

        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)

        np.random.seed(1)
        return self._test_residuals_fit(
            n_cells=n_cells,
            n_genes=n_genes,
            noise_model="norm"
        )


if __name__ == '__main__':
    unittest.main()
