import unittest
import logging
import numpy as np
import pandas as pd
import scipy.stats as stats

import diffxpy.api as de


class _TestSingleDE:

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
            from batchglm.api.models.glm_nb import Simulator
        elif noise_model == "norm":
            from batchglm.api.models.glm_norm import Simulator
        else:
            raise ValueError("noise model %s not recognized" % noise_model)

        num_non_de = n_genes // 2
        sim = Simulator(num_observations=n_cells, num_features=n_genes)
        sim.generate_sample_description(num_batches=0, num_conditions=2)
        sim.generate_params(
            rand_fn_ave=lambda shape: np.random.poisson(500, shape) + 1,
            rand_fn=lambda shape: np.abs(np.random.uniform(1, 0.5, shape))
        )
        sim.params["a_var"][1, :num_non_de] = 0
        sim.params["b_var"][1, :num_non_de] = 0
        sim.params["isDE"] = ("features",), np.arange(n_genes) >= num_non_de
        sim.generate_data()

        return sim

    def _eval(self, sim, test):
        idx_de = np.where(sim.params["isDE"] == True)[0]
        idx_nonde = np.where(sim.params["isDE"] == False)[0]

        frac_de_of_non_de = np.sum(test.qval[idx_nonde] < 0.05) / len(idx_nonde)
        frac_de_of_de = np.sum(test.qval[idx_de] < 0.05) / len(idx_de)

        logging.getLogger("diffxpy").info(
            'fraction of non-DE genes with q-value < 0.05: %.1f%%' %
            float(100 * frac_de_of_non_de)
        )
        logging.getLogger("diffxpy").info(
            'fraction of DE genes with q-value < 0.05: %.1f%%' %
            float(100 * frac_de_of_de)
        )
        assert frac_de_of_non_de <= 0.1, "too many false-positives"
        assert frac_de_of_de >= 0.5, "too many false-negatives"

        return sim

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

        sim = self._prepare_data(
            n_cells=n_cells,
            n_genes=n_genes,
            noise_model="norm"
        )

        test = de.test.rank_test(
            data=sim.X,
            grouping="condition",
            sample_description=sim.sample_description,
            dtype="float64"
        )

        self._eval(sim=sim, test=test)

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

        sim = self._prepare_data(
            n_cells=n_cells,
            n_genes=n_genes,
            noise_model="norm"
        )

        test = de.test.t_test(
            data=sim.X,
            grouping="condition",
            sample_description=sim.sample_description,
            dtype="float64"
        )

        self._eval(sim=sim, test=test)

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

        sim = self._prepare_data(
            n_cells=n_cells,
            n_genes=n_genes,
            noise_model=noise_model
        )

        test = de.test.wald(
            data=sim.X,
            factor_loc_totest="condition",
            formula_loc="~ 1 + condition",
            sample_description=sim.sample_description,
            noise_model=noise_model,
            training_strategy="DEFAULT",
            dtype="float64"
        )

        self._eval(sim=sim, test=test)

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

        sim = self._prepare_data(
            n_cells=n_cells,
            n_genes=n_genes,
            noise_model=noise_model
        )

        test = de.test.lrt(
            data=sim.X,
            full_formula_loc="~ 1 + condition",
            full_formula_scale="~ 1",
            reduced_formula_loc="~ 1",
            reduced_formula_scale="~ 1",
            sample_description=sim.sample_description,
            noise_model=noise_model,
            training_strategy="DEFAULT",
            dtype="float64"
        )

        self._eval(sim=sim, test=test)

        return True


class TestSingleDE_STANDARD(_TestSingleDE, unittest.TestCase):
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
        return self._test_rank_de(
            n_cells=n_cells,
            n_genes=n_genes
        )


class TestSingleDE_NB(_TestSingleDE, unittest.TestCase):
    """
    Negative binomial noise model unit tests that tests false positive and false negative rates.
    """

    def test_wald_de_nb(
            self,
            n_cells: int = 2000,
            n_genes: int = 200
    ):
        """
        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        """
        return self._test_wald_de(
            n_cells=n_cells,
            n_genes=n_genes,
            noise_model="nb"
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
        return self._test_lrt_de(
            n_cells=n_cells,
            n_genes=n_genes,
            noise_model="nb"
        )


class TestSingleDE_NORM(_TestSingleDE, unittest.TestCase):
    """
    Normal noise model unit tests that tests false positive and false negative rates.
    """

    def test_wald_de_norm(
            self,
            n_cells: int = 2000,
            n_genes: int = 200
    ):
        """
        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        """
        return self._test_wald_de(
            n_cells=n_cells,
            n_genes=n_genes,
            noise_model="norm"
        )

    def test_lrt_de_norm(
            self,
            n_cells: int = 2000,
            n_genes: int = 200
    ):
        """
        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        """
        return self._test_lrt_de(
            n_cells=n_cells,
            n_genes=n_genes,
            noise_model="norm"
        )


if __name__ == '__main__':
    unittest.main()
