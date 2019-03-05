import unittest
import logging
import numpy as np
import pandas as pd
import scipy.stats as stats

from batchglm.api.models.glm_nb import Simulator
import diffxpy.api as de


class TestSingleNull(unittest.TestCase):

    def test_null_distribution_wald(self, n_cells: int = 2000, n_genes: int = 100):
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
            "condition": np.random.randint(2, size=sim.num_observations),
            "batch": np.random.randint(2, size=sim.num_observations)
        })

        test = de.test.wald(
            data=sim.X,
            factor_loc_totest="condition",
            formula_loc="~ 1 + condition + batch",
            sample_description=random_sample_description,
            batch_size=500,
            training_strategy="DEFAULT",
            dtype="float64"
        )
        summary = test.summary()

        # Compare p-value distribution under null model against uniform distribution.
        pval_h0 = stats.kstest(test.pval, 'uniform').pvalue

        logging.getLogger("diffxpy").info('KS-test pvalue for null model match of wald(): %f' % pval_h0)
        assert pval_h0 > 0.05, "KS-Test failed: pval_h0=%f is <= 0.05!" % np.round(pval_h0, 5)

        return True

    def test_null_distribution_wald_multi(self, n_cells: int = 2000, n_genes: int = 100):
        """
        Test if de.wald() (multivariate mode) generates a uniform p-value distribution
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
            "condition": np.random.randint(4, size=sim.num_observations)
        })

        test = de.test.wald(
            data=sim.X,
            factor_loc_totest="condition",
            formula_loc="~ 1 + condition",
            sample_description=random_sample_description,
            training_strategy="DEFAULT",
            dtype="float64"
        )
        summary = test.summary()

        # Compare p-value distribution under null model against uniform distribution.
        pval_h0 = stats.kstest(test.pval, 'uniform').pvalue

        logging.getLogger("diffxpy").info('KS-test pvalue for null model match of wald(): %f' % pval_h0)
        assert pval_h0 > 0.05, "KS-Test failed: pval_h0=%f is <= 0.05!" % np.round(pval_h0, 5)

        return True

    def test_null_distribution_lrt(self, n_cells: int = 2000, n_genes: int = 100):
        """
        Test if de.lrt() generates a uniform p-value distribution
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

        test = de.test.lrt(
            data=sim.X,
            full_formula_loc="~ 1 + condition",
            full_formula_scale="~ 1",
            reduced_formula_loc="~ 1",
            reduced_formula_scale="~ 1",
            sample_description=random_sample_description,
            training_strategy="DEFAULT",
            dtype="float64"
        )
        summary = test.summary()

        # Compare p-value distribution under null model against uniform distribution.
        pval_h0 = stats.kstest(test.pval, 'uniform').pvalue

        logging.getLogger("diffxpy").info('KS-test pvalue for null model match of lrt(): %f' % pval_h0)
        assert pval_h0 > 0.05, "KS-Test failed: pval_h0=%f is <= 0.05!" % np.round(pval_h0, 5)

        return True

    def test_null_distribution_ttest(self, n_cells: int = 2000, n_genes: int = 100):
        """
        Test if de.t_test() generates a uniform p-value distribution
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

        test = de.test.t_test(
            data=sim.X,
            grouping="condition",
            sample_description=random_sample_description,
            is_logged=False,
            dtype="float64"
        )
        summary = test.summary()

        # Compare p-value distribution under null model against uniform distribution.
        pval_h0 = stats.kstest(test.pval, 'uniform').pvalue
        print(test.pval)

        logging.getLogger("diffxpy").info('KS-test pvalue for null model match of t_test(): %f' % pval_h0)
        assert pval_h0 > 0.05, "KS-Test failed: pval_h0=%f is <= 0.05!" % np.round(pval_h0, 5)

        return True

    def test_null_distribution_wilcoxon(self, n_cells: int = 2000, n_genes: int = 100):
        """
        Test if de.wilcoxon() generates a uniform p-value distribution
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

        test = de.test.rank_test(
            data=sim.X,
            grouping="condition",
            sample_description=random_sample_description,
            dtype="float64"
        )
        summary = test.summary()

        # Compare p-value distribution under null model against uniform distribution.
        pval_h0 = stats.kstest(test.pval, 'uniform').pvalue

        logging.getLogger("diffxpy").info('KS-test pvalue for null model match of wilcoxon(): %f' % pval_h0)
        assert pval_h0 > 0.05, "KS-Test failed: pval_h0=%f is <= 0.05!" % np.round(pval_h0, 5)

        return True


class TestSingleDE(unittest.TestCase):

    def _prepare_data(self, n_cells: int = 2000, n_genes: int = 100):
        """

        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        """
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

    def test_wilcoxon_de(self, n_cells: int = 2000, n_genes: int = 100):
        """
        Test if de.test.t_test() generates a uniform p-value distribution
        if it is given data simulated based on the null model.

        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)

        sim = self._prepare_data(n_cells=n_cells, n_genes=n_genes)

        test = de.test.rank_test(
            data=sim.X,
            grouping="condition",
            sample_description=sim.sample_description,
            dtype="float64"
        )

        self._eval(sim=sim, test=test)

        return True

    def test_t_test_de(self, n_cells: int = 2000, n_genes: int = 100):
        """
        Test if de.test.t_test() generates a uniform p-value distribution
        if it is given data simulated based on the null model.

        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)

        sim = self._prepare_data(n_cells=n_cells, n_genes=n_genes)

        test = de.test.t_test(
            data=sim.X,
            grouping="condition",
            sample_description=sim.sample_description,
            dtype="float64"
        )

        self._eval(sim=sim, test=test)

        return True

    def test_wald_de(self, n_cells: int = 2000, n_genes: int = 100):
        """
        Test if de.test.wald() generates a uniform p-value distribution
        if it is given data simulated based on the null model.

        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)

        sim = self._prepare_data(n_cells=n_cells, n_genes=n_genes)

        test = de.test.wald(
            data=sim.X,
            factor_loc_totest="condition",
            formula_loc="~ 1 + condition",
            sample_description=sim.sample_description,
            training_strategy="DEFAULT",
            dtype="float64"
        )

        self._eval(sim=sim, test=test)

        return True

    def test_lrt_de(self, n_cells: int = 2000, n_genes: int = 100):
        """
        Test if de.test.lrt() generates a uniform p-value distribution
        if it is given data simulated based on the null model. Returns the p-value
        of the two-side Kolmgorov-Smirnov test for equality of the observed
        p-value distribution and a uniform distribution.

        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)

        sim = self._prepare_data(n_cells=n_cells, n_genes=n_genes)

        test = de.test.lrt(
            data=sim.X,
            full_formula_loc="~ 1 + condition",
            full_formula_scale="~ 1",
            reduced_formula_loc="~ 1",
            reduced_formula_scale="~ 1",
            sample_description=sim.sample_description,
            training_strategy="DEFAULT",
            dtype="float64"
        )

        self._eval(sim=sim, test=test)

        return True


class TestSingleExternal(unittest.TestCase):

    def _prepare_data(self, n_cells: int = 2000, n_genes: int = 100):
        """

        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        """
        sim = Simulator(num_observations=n_cells, num_features=n_genes)
        sim.generate_sample_description(num_batches=0, num_conditions=2)
        sim.generate_params(
            rand_fn_ave=lambda shape: np.random.poisson(500, shape) + 1,
            rand_fn=lambda shape: np.abs(np.random.uniform(1, 0.5, shape))
        )
        sim.generate_data()

        return sim

    def _eval(self, test, ref_pvals):
        test_pval = test.pval
        pval_dev = np.abs(test_pval - ref_pvals)
        log_pval_dev = np.abs(np.log(test_pval+1e-200) - np.log(ref_pvals+1e-200))
        max_dev = np.max(pval_dev)
        max_log_dev = np.max(log_pval_dev)
        mean_dev = np.mean(log_pval_dev)
        logging.getLogger("diffxpy").info(
            'maximum absolute p-value deviation: %f' %
            float(max_dev)
        )
        logging.getLogger("diffxpy").info(
            'maximum absolute log p-value deviation: %f' %
            float(max_log_dev)
        )
        logging.getLogger("diffxpy").info(
            'mean absolute log p-value deviation: %f' %
            float(mean_dev)
        )
        assert max_dev < 1e-3, "maximum deviation too large"
        assert max_log_dev < 1e-1, "maximum deviation in log space too large"

    def test_t_test_ref(self, n_cells: int = 2000, n_genes: int = 100):
        """
        Test if de.test.t_test() generates the same p-value distribution as scipy t-test.

        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.INFO)

        sim = self._prepare_data(n_cells=n_cells, n_genes=n_genes)

        test = de.test.t_test(
            data=sim.X,
            grouping="condition",
            sample_description=sim.sample_description,
            dtype="float64"
        )

        # Run scipy t-tests as a reference.
        conds = np.unique(sim.sample_description["condition"].values)
        ind_a = np.where(sim.sample_description["condition"] == conds[0])[0]
        ind_b = np.where(sim.sample_description["condition"] == conds[1])[0]
        scipy_pvals = stats.ttest_ind(a=sim.X[ind_a, :], b=sim.X[ind_b, :], axis=0, equal_var=False).pvalue

        self._eval(test=test, ref_pvals=scipy_pvals)

        return True

    def test_wilcoxon_ref(self, n_cells: int = 2000, n_genes: int = 100):
        """
        Test if de.test.t_test() generates the same p-value distribution as scipy t-test.

        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.INFO)

        sim = self._prepare_data(n_cells=n_cells, n_genes=n_genes)

        test = de.test.rank_test(
            data=sim.X,
            grouping="condition",
            sample_description=sim.sample_description,
            dtype="float64"
        )

        # Run scipy t-tests as a reference.
        conds = np.unique(sim.sample_description["condition"].values)
        ind_a = np.where(sim.sample_description["condition"] == conds[0])[0]
        ind_b = np.where(sim.sample_description["condition"] == conds[1])[0]
        scipy_pvals = np.array([
            stats.mannwhitneyu(x=sim.X[ind_a, i], y=sim.X[ind_b, i],
                               use_continuity=True, alternative="two-sided").pvalue
            for i in range(sim.X.shape[1])
            ])

        self._eval(test=test, ref_pvals=scipy_pvals)

        return True


if __name__ == '__main__':
    unittest.main()
