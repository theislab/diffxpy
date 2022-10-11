import unittest
import logging
import numpy as np

import diffxpy.api as de
from batchglm.models.glm_nb import Model as NBModel
from batchglm.models.glm_norm import Model as NormModel
from batchglm.models.glm_poisson import Model as PoissonModel


class _TestSingleDe:

    def _prepare_data(
            self,
            n_cells: int,
            n_genes: int,
            noise_model: str
    ):
        """

        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        :param noise_model: Noise model to use for data fitting.
        """
        if noise_model == "nb":
            rand_fn_loc = lambda shape: np.random.uniform(5, 10, shape)
            rand_fn_scale = lambda shape: np.random.uniform(1, 2, shape)
            model = NBModel()
        elif noise_model == "norm":
            rand_fn_loc = lambda shape: np.random.uniform(500, 1000, shape)
            rand_fn_scale = lambda shape: np.random.uniform(1, 2, shape)
            model = NormModel()
        elif noise_model == "poisson":
            rand_fn_loc = lambda shape: np.random.uniform(2, 10, shape)
            rand_fn_scale = None # not used
            model = PoissonModel()
        else:
            raise ValueError("noise model %s not recognized" % noise_model)

        num_non_de = n_genes // 2
        def theta_location_setter(x):
            x[1, :num_non_de] = 0
            return x
        def theta_scale_setter(x):
            x[1, :num_non_de] = 0
            return x
        model.generate_artificial_data(
            n_obs=n_cells,
            n_vars=n_genes,
            num_batches=0,
            num_conditions=2,
            rand_fn_loc=rand_fn_loc,
            rand_fn_scale=rand_fn_scale,
            theta_location_setter=theta_location_setter,
            theta_scale_setter=theta_scale_setter,
        )
        self.isDE = np.arange(n_genes) >= num_non_de
        return model

    def _eval(self, model, test):
        idx_de = np.where(self.isDE)[0]
        idx_nonde = np.where(np.logical_not(self.isDE))[0]

        frac_de_of_non_de = np.mean(test.qval[idx_nonde] < 0.05)
        frac_de_of_de = np.mean(test.qval[idx_de] < 0.05)

        logging.getLogger("diffxpy").info(
            'fraction of non-DE genes with q-value < 0.05: %.1f%%' %
            float(100 * frac_de_of_non_de)
        )
        logging.getLogger("diffxpy").info(
            'fraction of DE genes with q-value < 0.05: %.1f%%' %
            float(100 * frac_de_of_de)
        )
        assert frac_de_of_non_de <= 0.1, "too many false-positives %f" % frac_de_of_non_de
        assert frac_de_of_de >= 0.5, "too many false-negatives %f" % frac_de_of_de

        return model

    def _test_rank_de(
            self,
            n_cells: int,
            n_genes: int
    ):
        """
        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)

        model = self._prepare_data(
            n_cells=n_cells,
            n_genes=n_genes,
            noise_model="norm"
        )

        test = de.test.rank_test(
            data=model.x,
            gene_names=model.features,
            sample_description=model.sample_description,
            grouping="condition"
        )

        self._eval(model=model, test=test)

        return True

    def _test_t_test_de(
            self,
            n_cells: int,
            n_genes: int
    ):
        """
        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)

        model = self._prepare_data(
            n_cells=n_cells,
            n_genes=n_genes,
            noise_model="norm"
        )

        test = de.test.t_test(
            data=model.x,
            gene_names=model.features,
            grouping="condition",
            sample_description=model.sample_description,
        )

        self._eval(model=model, test=test)

        return True

    def _test_wald_de(
            self,
            n_cells: int,
            n_genes: int,
            noise_model: str
    ):
        """
        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        :param noise_model: Noise model to use for data fitting.
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)

        model = self._prepare_data(
            n_cells=n_cells,
            n_genes=n_genes,
            noise_model=noise_model
        )

        test = de.test.wald(
            data=model.x,
            gene_names=model.features,
            sample_description=model.sample_description,
            factor_loc_totest="condition",
            formula_loc="~ 1 + condition",
            noise_model=noise_model,
            training_strategy="DEFAULT",
            dtype="float64"
        )

        self._eval(model=model, test=test)

        return True

    def _test_wald_repeated_de(
            self,
            n_cells: int,
            n_genes: int,
            noise_model: str
    ):
        """
        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        :param noise_model: Noise model to use for data fitting.
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)

        model = self._prepare_data(
            n_cells=n_cells,
            n_genes=n_genes,
            noise_model=noise_model
        )

        test1 = de.test.wald(
            data=model.x,
            gene_names=model.features,
            sample_description=model.sample_description,
            factor_loc_totest="condition",
            formula_loc="~ 1 + condition",
            noise_model=noise_model,
            training_strategy="DEFAULT",
            dtype="float64"
        )
        test = de.test.wald_repeated(
            det=test1,
            factor_loc_totest="condition"
        )
        assert np.max(test.log10_pval_clean() - test1.log10_pval_clean()) < 1e-10

        self._eval(model=model, test=test)
        return True

    def _test_lrt_de(
            self,
            n_cells: int,
            n_genes: int,
            noise_model: str
    ):
        """
        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        :param noise_model: Noise model to use for data fitting.
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)

        model = self._prepare_data(
            n_cells=n_cells,
            n_genes=n_genes,
            noise_model=noise_model
        )

        test = de.test.lrt(
            data=model.x,
            gene_names=model.features,
            sample_description=model.sample_description,
            full_formula_loc="~ 1 + condition",
            full_formula_scale="~ 1",
            reduced_formula_loc="~ 1",
            reduced_formula_scale="~ 1",
            noise_model=noise_model,
            training_strategy="DEFAULT",
            dtype="float64"
        )

        self._eval(model=model, test=test)

        return True


class TestSingleDeStandard(_TestSingleDe, unittest.TestCase):
    """
    Noise model-independent tests unit tests that tests false positive and false negative rates.
    """

    def test_ttest_de(
            self,
            n_cells: int = 2000,
            n_genes: int = 200
    ):
        """
        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)

        np.random.seed(1)
        return self._test_t_test_de(
            n_cells=n_cells,
            n_genes=n_genes
        )

    def test_rank_de(
            self,
            n_cells: int = 2000,
            n_genes: int = 200
    ):
        """
        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)

        np.random.seed(1)
        return self._test_rank_de(
            n_cells=n_cells,
            n_genes=n_genes
        )


class TestSingleDeNb(_TestSingleDe, unittest.TestCase):

    noise_model = 'nb'

    """
    Negative binomial (default) noise model unit tests that tests false positive and false negative rates.
    """

    def test_wald_de(
            self,
            n_cells: int = 2000,
            n_genes: int = 200
    ):
        """
        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)

        np.random.seed(1)
        return self._test_wald_de(
            n_cells=n_cells,
            n_genes=n_genes,
            noise_model=self.noise_model
        )

    def test_wald_repeated_de_nb(
            self,
            n_cells: int = 2000,
            n_genes: int = 200
    ):
        """
        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)

        np.random.seed(1)
        return self._test_wald_repeated_de(
            n_cells=n_cells,
            n_genes=n_genes,
            noise_model=self.noise_model
        )

    def test_lrt_de_nb(
            self,
            n_cells: int = 2000,
            n_genes: int = 200
    ):
        """
        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)

        np.random.seed(1)
        return self._test_lrt_de(
            n_cells=n_cells,
            n_genes=n_genes,
            noise_model=self.noise_model
        )

class TestSingleDePoisson(TestSingleDeNb, unittest.TestCase):
    noise_model = "poisson"

class TestSingleDeNorm(TestSingleDeNb, unittest.TestCase):
    noise_model = "norm"


if __name__ == '__main__':
    unittest.main()
