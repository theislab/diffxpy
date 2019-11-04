import unittest
import logging
import numpy as np
import scipy.stats as stats

from batchglm.api.models.numpy.glm_nb import Simulator
import diffxpy.api as de


class TestSingleExternalLibs(unittest.TestCase):

    def _prepare_data(self, n_cells: int = 2000, n_genes: int = 100):
        """

        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        """
        sim = Simulator(num_observations=n_cells, num_features=n_genes)
        sim.generate_sample_description(num_batches=0, num_conditions=2)
        sim.generate_params()
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
        assert max_dev < 1e-3, "maximum deviation too large: %f" % max_dev
        assert max_log_dev < 1e-1, "maximum deviation in log space too large: %f" % max_log_dev

    def test_t_test_ref(self, n_cells: int = 2000, n_genes: int = 100):
        """
        Test if de.test.t_test() generates the same p-value distribution as scipy t-test.

        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.INFO)

        np.random.seed(1)
        sim = self._prepare_data(n_cells=n_cells, n_genes=n_genes)
        test = de.test.t_test(
            data=sim.input_data,
            grouping="condition",
            sample_description=sim.sample_description
        )

        # Run scipy t-tests as a reference.
        conds = np.unique(sim.sample_description["condition"].values)
        ind_a = np.where(sim.sample_description["condition"] == conds[0])[0]
        ind_b = np.where(sim.sample_description["condition"] == conds[1])[0]
        scipy_pvals = stats.ttest_ind(
            a=sim.x[ind_a, :],
            b=sim.x[ind_b, :],
            axis=0,
            equal_var=False
        ).pvalue
        self._eval(test=test, ref_pvals=scipy_pvals)
        return True

    def test_rank_ref(self, n_cells: int = 2000, n_genes: int = 100):
        """
        Test if de.test.rank_test() generates the same p-value distribution as scipy t-test.

        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.INFO)

        np.random.seed(1)
        sim = self._prepare_data(n_cells=n_cells, n_genes=n_genes)
        test = de.test.rank_test(
            data=sim.input_data,
            grouping="condition",
            sample_description=sim.sample_description
        )

        # Run scipy t-tests as a reference.
        conds = np.unique(sim.sample_description["condition"].values)
        ind_a = np.where(sim.sample_description["condition"] == conds[0])[0]
        ind_b = np.where(sim.sample_description["condition"] == conds[1])[0]
        scipy_pvals = np.array([
            stats.mannwhitneyu(
                x=sim.x[ind_a, i],
                y=sim.x[ind_b, i],
                use_continuity=True,
                alternative="two-sided"
            ).pvalue
            for i in range(sim.x.shape[1])
        ])
        self._eval(test=test, ref_pvals=scipy_pvals)
        return True


if __name__ == '__main__':
    unittest.main()
