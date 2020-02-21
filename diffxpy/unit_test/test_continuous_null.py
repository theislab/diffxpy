import unittest

import numpy as np
import pandas as pd
import scipy.stats as stats
import logging

from batchglm.api.models.numpy.glm_nb import Simulator
import diffxpy.api as de


class _TestContinuous:

    noise_model: str

    def _fit_continuous(
        self,
        sim,
        sample_description,
        constrained,
        test,
        spline_basis
    ):
        test = de.test.continuous_1d(
            data=sim.input_data,
            sample_description=sample_description,
            gene_names=["gene" + str(i) for i in range(sim.input_data.num_features)],
            formula_loc="~ 1 + continuous + batch" if constrained else "~ 1 + continuous",
            formula_scale="~ 1",
            factor_loc_totest="continuous",
            continuous="continuous",
            constraints_loc={"batch": "continuous"} if constrained else None,
            size_factors="size_factors",
            df=3,
            spline_basis=spline_basis,
            test=test,
            quick_scale=False,
            noise_model=self.noise_model
        )
        return test

    def _fit_continuous_interaction(
        self,
        sim,
        sample_description,
        constrained,
        test,
        spline_basis
    ):
        test = de.test.continuous_1d(
            data=sim.input_data,
            sample_description=sample_description,
            gene_names=["gene" + str(i) for i in range(sim.input_data.num_features)],
            formula_loc="~ 1 + continuous + condition + continuous:condition" if not constrained else \
                "~ 1 + continuous + condition + continuous:condition + batch",
            formula_scale="~ 1",
            factor_loc_totest=["continuous", "continuous:condition"],
            continuous="continuous",
            constraints_loc={"batch": "condition"} if constrained else None,
            size_factors="size_factors",
            df=3,
            spline_basis=spline_basis,
            test=test,
            quick_scale=False,
            noise_model=self.noise_model
        )
        return test

    def _test_basic(
            self,
            ngenes: int,
            test: str,
            constrained: bool,
            spline_basis: str
    ):
        n_timepoints = 5
        sim = Simulator(num_observations=n_timepoints*200, num_features=ngenes)
        sim.generate_sample_description(num_batches=0, num_conditions=0)
        sim.generate_params()
        sim.generate_data()

        random_sample_description = pd.DataFrame({
            "continuous": np.asarray(np.random.randint(0, n_timepoints, size=sim.nobs), dtype=float)
        })
        random_sample_description["batch"] = [str(int(x)) + str(np.random.randint(0, 3))
                                              for x in random_sample_description["continuous"]]
        random_sample_description["size_factors"] = np.random.uniform(0.9, 1.1, sim.nobs)  # TODO put into simulation.
        det = self._fit_continuous(
            sim=sim,
            sample_description=random_sample_description,
            test=test,
            constrained=constrained,
            spline_basis=spline_basis,
        )
        return det

    def _test_interaction(
            self,
            ngenes: int,
            test: str,
            constrained: bool,
            spline_basis: str
    ):
        n_timepoints = 5
        sim = Simulator(num_observations=n_timepoints*200, num_features=ngenes)
        sim.generate_sample_description(num_batches=0, num_conditions=0)
        sim.generate_params()
        sim.generate_data()

        random_sample_description = pd.DataFrame({
            "continuous": np.asarray(np.random.randint(0, n_timepoints, size=sim.nobs), dtype=float)
        })
        random_sample_description["condition"] = [str(np.random.randint(0, 2))
                                                  for x in random_sample_description["continuous"]]
        random_sample_description["batch"] = [x + str(np.random.randint(0, 3))
                                              for x in random_sample_description["condition"]]
        random_sample_description["size_factors"] = np.random.uniform(0.9, 1.1, sim.nobs)  # TODO put into simulation.
        det = self._fit_continuous_interaction(
            sim=sim,
            sample_description=random_sample_description,
            test=test,
            constrained=constrained,
            spline_basis=spline_basis,
        )
        return det

    def _test_null_model(
            self,
            ngenes: int,
            test: str,
            constrained: bool,
            spline_basis: str
    ):
        det = self._test_basic(
            ngenes=ngenes,
            test=test,
            constrained=constrained,
            spline_basis=spline_basis
        )
        return self._eval(det=det)

    def _test_null_model_interaction(
            self,
            ngenes: int,
            test: str,
            constrained: bool,
            spline_basis: str
    ):
        det = self._test_interaction(
            ngenes=ngenes,
            test=test,
            constrained=constrained,
            spline_basis=spline_basis
        )
        return self._eval(det=det)

    def _eval(self, det):
        pval_h0 = stats.kstest(det.pval, 'uniform').pvalue
        logging.getLogger("diffxpy").info(
            'KS-test pvalue for null model match of wald(): %f' % pval_h0
        )
        assert pval_h0 > 0.05, "KS-Test failed: pval_h0=%f is <= 0.05" % pval_h0
        return True

    def _test_forfatal(
            self,
            test: str,
            constrained: bool,
            spline_basis: str
    ):
        """
        Test if de.test.continuous() DifferentialExpressionTestSingle object functions work fine.
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)

        test = self._test_basic(
            ngenes=2,
            test=test,
            constrained=constrained,
            spline_basis=spline_basis
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

    def _test_for_fatal_all_splines(
            self,
            test: str,
            constrained: bool
    ):
        for x in ["bs", "cr", "cc"]:
            self._test_forfatal(test=test, constrained=constrained, spline_basis=x)

    def _test_null_model_all_splines(
            self,
            ngenes: int,
            test: str,
            constrained: bool
    ):
        for x in ["bs", "cr", "cc"]:
            self._test_null_model(ngenes=ngenes, test=test, constrained=constrained, spline_basis=x)

    def _test_null_model_all_splines_interaction(
            self,
            ngenes: int,
            test: str,
            constrained: bool
    ):
        for x in ["bs", "cr", "cc"]:
            self._test_null_model_interaction(ngenes=ngenes, test=test, constrained=constrained, spline_basis=x)


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
        self._test_for_fatal_all_splines(test="wald", constrained=False)
        self._test_for_fatal_all_splines(test="wald", constrained=True)
        return True

    def _test_forfatal_lrt(self):
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

    def test_null_distribution_wald_unconstrained(self):
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
        self._test_null_model_all_splines(ngenes=100, test="wald", constrained=False)
        self._test_null_model_all_splines_interaction(ngenes=100, test="wald", constrained=False)
        return True

    def test_null_distribution_wald_constrained(self):
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
        self._test_null_model_all_splines(ngenes=100, test="wald", constrained=True)
        self._test_null_model_all_splines_interaction(ngenes=100, test="wald", constrained=True)
        return True

    def _test_null_distribution_lrt(self):
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
        return True


if __name__ == '__main__':
    unittest.main()
