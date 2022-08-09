import logging
import unittest
import numpy as np
import pandas as pd
import scipy.stats as stats
from batchglm.models.glm_nb import Model as NBModel
from batchglm.models.glm_norm import Model as NormModel
from batchglm.models.glm_poisson import Model as PoissonModel

import diffxpy.api as de


class _TestPairwiseNull:

    noise_model: str

    def _prepate_data(
            self,
            n_cells: int,
            n_genes: int,
            n_groups: int
    ):


        if self.noise_model == "nb":
            rand_fn_loc = lambda shape: np.random.uniform(0.1, 1, shape)
            rand_fn_scale = lambda shape: np.random.uniform(0.5, 1, shape)
            model = NBModel()
        elif self.noise_model == "norm" or self.noise_model is None:
            rand_fn_loc = lambda shape: np.random.uniform(500, 1000, shape)
            rand_fn_scale = lambda shape: np.random.uniform(1, 2, shape)
            model = NormModel()
        elif self.noise_model == "poisson":
            rand_fn_loc = lambda shape: np.random.uniform(2, 10, shape)
            rand_fn_scale = None
            model = PoissonModel()
        else:
            raise ValueError("noise model %s not recognized" % self.noise_model)


        model.generate_artificial_data(
            n_obs=n_cells,
            n_vars=n_genes,
            num_batches=0,
            num_conditions=0,
            rand_fn_loc=rand_fn_loc,
            rand_fn_scale=rand_fn_scale
        )

        random_sample_description = pd.DataFrame({
            "condition": [str(x) for x in np.random.randint(n_groups, size=n_cells)]
        })
        return model, random_sample_description

    def _test_null_distribution_basic(
            self,
            test: str,
            lazy: bool,
            quick_scale: bool = False,
            n_cells: int = 3000,
            n_genes: int = 200,
            n_groups: int = 3
    ):
        """
        Test if de.wald() generates a uniform p-value distribution
        if it is given data simulated based on the null model. Returns the p-value
        of the two-side Kolmgorov-Smirnov test for equality of the observed 
        p-value distriubution and a uniform distribution.

        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        """
        model, sample_description = self._prepate_data(
            n_cells=n_cells,
            n_genes=n_genes,
            n_groups=n_groups
        )
        det = de.test.pairwise(
            data=model.x,
            gene_names=model.features,
            sample_description=sample_description,
            grouping="condition",
            test=test,
            lazy=lazy,
            quick_scale=quick_scale,
            noise_model=self.noise_model
        )
        if not lazy:
            _ = det.summary()
            _ = det.pval
            _ = det.qval
            _ = det.log_fold_change()
        # Single pair accessors:
        _ = det.pval_pairs(groups0="0", groups1="1")
        _ = det.qval_pairs(groups0="0", groups1="1")
        _ = det.log10_pval_pairs_clean(groups0="0", groups1="1")
        _ = det.log10_qval_pairs_clean(groups0="0", groups1="1")
        _ = det.log_fold_change_pairs(groups0="0", groups1="1")
        _ = det.summary_pairs(groups0="0", groups1="1")

        # Compare p-value distribution under null model against uniform distribution.
        pval_h0 = stats.kstest(det.pval_pairs(groups0="0", groups1="1").flatten(), 'uniform').pvalue

        logging.getLogger("diffxpy").info('KS-test pvalue for null model match of wald(): %f' % pval_h0)
        assert pval_h0 > 0.05, "KS-Test failed: pval_h0=%f is <= 0.05!" % np.round(pval_h0, 5)
        return True


class TestPairwiseNullStandard(unittest.TestCase, _TestPairwiseNull):

    def test_null_distribution_ttest(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)

        np.random.seed(1)
        self.noise_model = "norm"
        self._test_null_distribution_basic(test="t-test", lazy=False)

    def test_null_distribution_rank(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)

        np.random.seed(1)
        self.noise_model = "norm"
        self._test_null_distribution_basic(test="rank", lazy=False)

class TestPairwiseNullPoisson(unittest.TestCase, _TestPairwiseNull):

    def test_null_distribution_ztest(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)

        np.random.seed(1)
        self.noise_model = "poisson"
        self._test_null_distribution_basic(test="z-test", lazy=False, quick_scale=False)

    def test_null_distribution_ztest_lazy(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)

        np.random.seed(1)
        self.noise_model = "poisson"
        self._test_null_distribution_basic(test="z-test", lazy=True, quick_scale=False)

    def test_null_distribution_wald(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)

        np.random.seed(1)
        self.noise_model = "poisson"
        self._test_null_distribution_basic(test="wald", lazy=False, quick_scale=False)

    def test_null_distribution_lrt(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)

        np.random.seed(1)
        self.noise_model = "poisson"
        self._test_null_distribution_basic(test="lrt", lazy=False, quick_scale=False)



class TestPairwiseNullNb(unittest.TestCase, _TestPairwiseNull):

    def test_null_distribution_ztest(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)

        np.random.seed(1)
        self.noise_model = "nb"
        self._test_null_distribution_basic(test="z-test", lazy=False, quick_scale=False)
        self._test_null_distribution_basic(test="z-test", lazy=False, quick_scale=True)

    def test_null_distribution_ztest_lazy(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)

        np.random.seed(1)
        self.noise_model = "nb"
        self._test_null_distribution_basic(test="z-test", lazy=True, quick_scale=False)
        self._test_null_distribution_basic(test="z-test", lazy=True, quick_scale=True)

    def test_null_distribution_wald(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)

        np.random.seed(1)
        self.noise_model = "nb"
        self._test_null_distribution_basic(test="wald", lazy=False, quick_scale=False)
        self._test_null_distribution_basic(test="wald", lazy=False, quick_scale=True)

    def test_null_distribution_lrt(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)

        np.random.seed(1)
        self.noise_model = "nb"
        self._test_null_distribution_basic(test="lrt", lazy=False, quick_scale=False)


if __name__ == '__main__':
    unittest.main()
