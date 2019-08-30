import unittest

import numpy as np
import pandas as pd
import scipy.stats as stats
import logging

from batchglm.api.models.glm_nb import Simulator
import diffxpy.api as de


class _TestContinuous:

    noise_model: str

    def _fit_continuous(
        self,
        sim,
        sample_description,
        constrained,
        test
    ):
        test = de.test.continuous_1d(
            data=sim.input_data,
            sample_description=sample_description,
            gene_names=["gene" + str(i) for i in range(sim.input_data.num_features)],
            formula_loc="~ 1 + continuous_covariate + batch",
            formula_scale="~ 1",
            factor_loc_totest="continuous_covariate",
            continuous="continuous_covariate",
            constraints_loc={"batch": "continuous_covariate"} if constrained else None,
            df=3,
            test=test,
            quick_scale=True,
            noise_model=self.noise_model
        )
        return test

    def _test_null_model(
            self,
            nobs: int,
            ngenes: int,
            test: str,
            constrained: bool
    ):
        sim = Simulator(num_observations=nobs, num_features=ngenes)
        sim.generate_sample_description(num_batches=0, num_conditions=0)
        sim.generate()

        random_sample_description = pd.DataFrame({
            "continuous_covariate": np.asarray(np.random.randint(0, 5, size=sim.nobs), dtype=float)
        })
        random_sample_description["batch"] = [str(int(x)) + str(np.random.randint(0, 3))
                                              for x in random_sample_description["continuous_covariate"]]
        return self._fit_continuous(
            sim=sim,
            sample_description=random_sample_description,
            test=test,
            constrained=constrained
        )

    def _test_forfatal(
            self,
            test: str,
            constrained: bool
    ):
        """
        Test if de.test.continuous() DifferentialExpressionTestSingle object functions work fine.
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)

        test = self._test_null_model(
            nobs=10,
            ngenes=2,
            test=test,
            constrained=constrained
        )
        ids = test.gene_ids

        # 1. Test all additional functions which depend on model computation:
        # 1.1. Only continuous model:
        _ = test.log_fold_change(genes=ids, non_numeric=False)
        _ = test.max(genes=ids, non_numeric=False)
        _ = test.min(genes=ids, non_numeric=False)
        _ = test.argmax(genes=ids, non_numeric=False)
        _ = test.argmin(genes=ids, non_numeric=False)
        _ = test.summary(non_numeric=False)
        # 1.2. Full model:
        _ = test.log_fold_change(genes=ids, non_numeric=True)
        _ = test.max(genes=ids, non_numeric=True)
        _ = test.min(genes=ids, non_numeric=True)
        _ = test.argmax(genes=ids, non_numeric=True)
        _ = test.argmin(genes=ids, non_numeric=True)
        _ = test.summary(non_numeric=True)

        return True


class TestContinuousNb(_TestContinuous, unittest.TestCase):

    def test_forfatal_wald(self):
        """
        Test if de.test.continuous() generates a uniform p-value distribution in the wald test
        if it is given data simulated based on the null model. Returns the p-value
        of the two-side Kolmgorov-Smirnov test for equality of the observed
        p-value distriubution and a uniform distribution.

        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)

        self.noise_model = "nb"
        np.random.seed(1)
        _ = self._test_forfatal(test="wald", constrained=False)
        _ = self._test_forfatal(test="wald", constrained=True)
        return True

    def test_forfatal_lrt(self):
        """
        Test if de.test.continuous() generates a uniform p-value distribution in the wald test
        if it is given data simulated based on the null model. Returns the p-value
        of the two-side Kolmgorov-Smirnov test for equality of the observed
        p-value distriubution and a uniform distribution.

        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)

        self.noise_model = "nb"
        np.random.seed(1)
        _ = self._test_forfatal(test="lrt", constrained=False)
        return True

    def test_null_distribution_wald(self):
        """
        Test if de.test.continuous() generates a uniform p-value distribution in the wald test
        if it is given data simulated based on the null model. Returns the p-value
        of the two-side Kolmgorov-Smirnov test for equality of the observed 
        p-value distriubution and a uniform distribution.

        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)

        self.noise_model = "nb"
        np.random.seed(1)
        test = self._test_null_model(nobs=2000, ngenes=100, test="wald", constrained=False)

        # Compare p-value distribution under null model against uniform distribution.
        pval_h0 = stats.kstest(test.pval, 'uniform').pvalue

        logging.getLogger("diffxpy").info('KS-test pvalue for null model match of wald(): %f' % pval_h0)
        assert pval_h0 > 0.05, "KS-Test failed: pval_h0 is <= 0.05!"

        return True

    def test_null_distribution_lrt(self):
        """
        Test if de.test.continuous() generates a uniform p-value distribution in lrt
        if it is given data simulated based on the null model. Returns the p-value
        of the two-side Kolmgorov-Smirnov test for equality of the observed
        p-value distriubution and a uniform distribution.

        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        """
        logging.getLogger("tensorflow").setLevel(logging.INFO)
        logging.getLogger("batchglm").setLevel(logging.INFO)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)

        self.noise_model = "nb"
        np.random.seed(1)
        test = self._test_null_model(nobs=2000, ngenes=100, test="lrt", constrained=False)

        # Compare p-value distribution under null model against uniform distribution.
        pval_h0 = stats.kstest(test.pval, 'uniform').pvalue

        logging.getLogger("diffxpy").info('KS-test pvalue for null model match of wald(): %f' % pval_h0)
        assert pval_h0 > 0.05, "KS-Test failed: pval_h0 is <= 0.05!"

        return True


if __name__ == '__main__':
    unittest.main()
