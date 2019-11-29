import unittest
import logging
import numpy as np
import pandas as pd
import scipy.stats as stats

import diffxpy.api as de


class _TestSingleNull:

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

        test = de.test.wald(
            data=sim.input_data,
            sample_description=random_sample_description,
            factor_loc_totest="condition",
            formula_loc="~ 1 + condition + batch",
            noise_model=noise_model
        )
        _ = test.summary()

        # Compare p-value distribution under null model against uniform distribution.
        pval_h0 = stats.kstest(test.pval, 'uniform').pvalue

        logging.getLogger("diffxpy").info('KS-test pvalue for null model match of wald(): %f' % pval_h0)
        assert pval_h0 > 0.05, ("KS-Test failed: pval_h0=%f is <= 0.05!" % np.round(pval_h0, 5))

        return True

    def _test_null_distribution_wald_repeated(
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

        test1 = de.test.wald(
            data=sim.input_data,
            sample_description=random_sample_description,
            factor_loc_totest="condition",
            formula_loc="~ 1 + condition + batch",
            noise_model=noise_model
        )
        test = de.test.wald_repeated(
            det=test1,
            factor_loc_totest="condition"
        )

        _ = test.summary()

        # Compare p-value distribution under null model against uniform distribution.
        pval_h0 = stats.kstest(test.pval, 'uniform').pvalue

        logging.getLogger("diffxpy").info('KS-test pvalue for null model match of wald_repeated(): %f' % pval_h0)
        assert pval_h0 > 0.05, ("KS-Test failed: pval_h0=%f is <= 0.05!" % np.round(pval_h0, 5))

        return True

    def _test_null_distribution_wald_multi(
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
            from batchglm.api.models.numpy.glm_nb import Simulator
        elif noise_model == "norm":
            from batchglm.api.models.numpy.glm_norm import Simulator
        else:
            raise ValueError("noise model %s not recognized" % noise_model)

        sim = Simulator(num_observations=n_cells, num_features=n_genes)
        sim.generate_sample_description(num_batches=0, num_conditions=0)
        sim.generate()

        random_sample_description = pd.DataFrame({
            "condition": np.random.randint(4, size=sim.nobs)
        })

        test = de.test.wald(
            data=sim.input_data,
            sample_description=random_sample_description,
            factor_loc_totest="condition",
            formula_loc="~ 1 + condition",
            noise_model=noise_model
        )
        _ = test.summary()

        # Compare p-value distribution under null model against uniform distribution.
        pval_h0 = stats.kstest(test.pval, 'uniform').pvalue

        logging.getLogger("diffxpy").info('KS-test pvalue for null model match of wald(): %f' % pval_h0)
        assert pval_h0 > 0.05, ("KS-Test failed: pval_h0=%f is <= 0.05!" % np.round(pval_h0, 5))

        return True

    def _test_null_distribution_lrt(
            self,
            n_cells: int,
            n_genes: int,
            noise_model: str
    ):
        """
        Test if de.lrt() generates a uniform p-value distribution
        if it is given data simulated based on the null model. Returns the p-value
        of the two-side Kolmgorov-Smirnov test for equality of the observed
        p-value distribution and a uniform distribution.

        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        :param noise_model: Noise model to use for data fitting.
        """
        if noise_model == "nb":
            from batchglm.api.models.numpy.glm_nb import Simulator
        elif noise_model == "norm":
            from batchglm.api.models.numpy.glm_norm import Simulator
        else:
            raise ValueError("noise model %s not recognized" % noise_model)

        sim = Simulator(num_observations=n_cells, num_features=n_genes)
        sim.generate_sample_description(num_batches=0, num_conditions=0)
        sim.generate()

        random_sample_description = pd.DataFrame({
            "condition": np.random.randint(2, size=sim.nobs)
        })

        test = de.test.lrt(
            data=sim.input_data,
            sample_description=random_sample_description,
            full_formula_loc="~ 1 + condition",
            full_formula_scale="~ 1",
            reduced_formula_loc="~ 1",
            reduced_formula_scale="~ 1",
            noise_model=noise_model
        )
        _ = test.summary()

        # Compare p-value distribution under null model against uniform distribution.
        pval_h0 = stats.kstest(test.pval, 'uniform').pvalue

        logging.getLogger("diffxpy").info('KS-test pvalue for null model match of lrt(): %f' % pval_h0)
        assert pval_h0 > 0.05, ("KS-Test failed: pval_h0=%f is <= 0.05!" % np.round(pval_h0, 5))

        return True

    def _test_null_distribution_ttest(
            self,
            n_cells: int,
            n_genes: int
    ):
        """
        Test if de.t_test() generates a uniform p-value distribution
        if it is given data simulated based on the null model. Returns the p-value
        of the two-side Kolmgorov-Smirnov test for equality of the observed
        p-value distribution and a uniform distribution.

        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        """
        from batchglm.api.models.numpy.glm_norm import Simulator

        sim = Simulator(num_observations=n_cells, num_features=n_genes)
        sim.generate_sample_description(num_batches=0, num_conditions=0)
        sim.generate()

        random_sample_description = pd.DataFrame({
            "condition": np.random.randint(2, size=sim.nobs)
        })

        test = de.test.t_test(
            data=sim.input_data,
            sample_description=random_sample_description,
            grouping="condition",
            is_logged=False
        )
        _ = test.summary()

        # Compare p-value distribution under null model against uniform distribution.
        pval_h0 = stats.kstest(test.pval, 'uniform').pvalue

        logging.getLogger("diffxpy").info('KS-test pvalue for null model match of t_test(): %f' % pval_h0)
        assert pval_h0 > 0.05, ("KS-Test failed: pval_h0=%f is <= 0.05!" % np.round(pval_h0, 5))

        return True

    def _test_null_distribution_rank(
            self,
            n_cells: int,
            n_genes: int
    ):
        """
        Test if de.test.rank_test() generates a uniform p-value distribution
        if it is given data simulated based on the null model. Returns the p-value
        of the two-side Kolmgorov-Smirnov test for equality of the observed
        p-value distribution and a uniform distribution.

        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        """
        from batchglm.api.models.numpy.glm_norm import Simulator

        sim = Simulator(num_observations=n_cells, num_features=n_genes)
        sim.generate_sample_description(num_batches=0, num_conditions=0)
        sim.generate()

        random_sample_description = pd.DataFrame({
            "condition": np.random.randint(2, size=sim.nobs)
        })

        test = de.test.rank_test(
            data=sim.input_data,
            sample_description=random_sample_description,
            grouping="condition"
        )
        _ = test.summary()

        # Compare p-value distribution under null model against uniform distribution.
        pval_h0 = stats.kstest(test.pval, 'uniform').pvalue

        logging.getLogger("diffxpy").info('KS-test pvalue for null model match of rank_test(): %f' % pval_h0)
        assert pval_h0 > 0.05, ("KS-Test failed: pval_h0=%f is <= 0.05!" % np.round(pval_h0, 5))

        return True


class TestSingleNullStandard(_TestSingleNull, unittest.TestCase):
    """
    Noise model-independent unit tests that test whether a test generates uniformly
    distributed p-values if data are sampled from the null model.
    """

    def test_null_distribution_ttest(
            self,
            n_cells: int = 2000,
            n_genes: int = 200
    ):
        """
        Test if t_test() generates a uniform p-value distribution.

        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)

        np.random.seed(1)
        return self._test_null_distribution_ttest(
            n_cells=n_cells,
            n_genes=n_genes
        )

    def test_null_distribution_rank(
            self,
            n_cells: int = 2000,
            n_genes: int = 200
    ):
        """
        Test if t_test() generates a uniform p-value distribution.

        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)

        np.random.seed(1)
        return self._test_null_distribution_rank(
            n_cells=n_cells,
            n_genes=n_genes
        )


class TestSingleNullNb(_TestSingleNull, unittest.TestCase):
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

    def test_null_distribution_wald_repeated_nb(
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
        return self._test_null_distribution_wald_repeated(
            n_cells=n_cells,
            n_genes=n_genes,
            noise_model="nb"
        )

    def test_null_distribution_wald_multi_nb(
            self,
            n_cells: int = 2000,
            n_genes: int = 200
    ):
        """
        Test if wald() generates a uniform p-value distribution for "nb" noise model
        for multiple coefficients to test.

        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)

        np.random.seed(1)
        return self._test_null_distribution_wald_multi(
            n_cells=n_cells,
            n_genes=n_genes,
            noise_model="nb"
        )

    def test_null_distribution_lrt_nb(
            self,
            n_cells: int = 2000,
            n_genes: int = 200
    ):
        """
        Test if lrt() generates a uniform p-value distribution for "nb" noise model.

        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)

        np.random.seed(1)
        return self._test_null_distribution_lrt(
            n_cells=n_cells,
            n_genes=n_genes,
            noise_model="nb"
        )


class TestSingleNullNorm(_TestSingleNull, unittest.TestCase):
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

    def test_null_distribution_wald_multi_norm(
            self,
            n_cells: int = 2000,
            n_genes: int = 200
    ):
        """
        Test if wald() generates a uniform p-value distribution for "norm" noise model
        for multiple coefficients to test.

        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)

        np.random.seed(1)
        return self._test_null_distribution_wald_multi(
            n_cells=n_cells,
            n_genes=n_genes,
            noise_model="norm"
        )

    def test_null_distribution_lrt_norm(
            self,
            n_cells: int = 2000,
            n_genes: int = 200
    ):
        """
        Test if lrt() generates a uniform p-value distribution for "norm" noise model.

        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)

        np.random.seed(1)
        return self._test_null_distribution_lrt(
            n_cells=n_cells,
            n_genes=n_genes,
            noise_model="norm"
        )


if __name__ == '__main__':
    unittest.main()
