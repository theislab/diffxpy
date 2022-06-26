import unittest
import logging
import numpy as np

import diffxpy.api as de

from batchglm.models.glm_nb import Model as NBModel
from batchglm.models.glm_norm import Model as NormModel

class _TestContinuousDe:
    noise_model: str

    def _test_wald_de(
            self,
            constrained: bool,
            spline_basis: str,
            ngenes: int
    ):
        if self.noise_model == "nb":
            model = NBModel()
            rand_fn_loc = lambda shape: np.random.uniform(2, 5, shape)
            rand_fn_scale = lambda shape: np.random.uniform(1, 2, shape)
        elif self.noise_model == "norm":
            model = NormModel()
            rand_fn_loc = lambda shape: np.random.uniform(500, 1000, shape)
            rand_fn_scale = lambda shape: np.random.uniform(1, 2, shape)
        else:
            raise ValueError("noise model %s not recognized" % self.noise_model)

        n_timepoints = 7

        num_non_de = round(ngenes / 2)
        def theta_location_setter(x):
            x[1:, :num_non_de] = 0
            return x
        def theta_scale_setter(x):
            x[1:, :] = 0
            return x
        model.generate_artificial_data(
            n_obs=n_timepoints*200,
            n_vars=ngenes,
            num_batches=0,
            num_conditions=n_timepoints,
            rand_fn_loc=rand_fn_loc,
            rand_fn_scale=rand_fn_scale,
            theta_location_setter=theta_location_setter,
            theta_scale_setter=theta_scale_setter
        )
        self.isDE = np.arange(ngenes) >= num_non_de

        random_sample_description = model.sample_description
        random_sample_description["continuous"] = [int(x) for x in random_sample_description["condition"]]
        random_sample_description["batch"] = [
            str(int(x)) + str(np.random.randint(0, 3))
            for x in random_sample_description["continuous"]
        ]

        test = de.test.continuous_1d(
            data=model.x,
            gene_names=model.features,
            sample_description=random_sample_description,
            formula_loc="~ 1 + continuous + batch" if constrained else "~ 1 + continuous",
            formula_scale="~ 1",
            factor_loc_totest="continuous",
            continuous="continuous",
            constraints_loc={"batch": "continuous"} if constrained else None,
            df=5,
            spline_basis=spline_basis,
            test="wald",
            quick_scale=True,
            noise_model=self.noise_model
        )
        self._eval(model=model, test=test)

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
        assert frac_de_of_non_de <= 0.1, "too many false-positives, FPR=%f" % frac_de_of_non_de
        assert frac_de_of_de >= 0.5, "too many false-negatives, TPR=%f" % frac_de_of_de

        return model

    def _test_wald_de_all_splines(
            self,
            ngenes: int,
            constrained: bool
    ):
        for x in ["bs", "cr", "cc"]:
            self._test_wald_de(ngenes=ngenes, constrained=constrained, spline_basis=x)


class TestContinuousDeNb(_TestContinuousDe, unittest.TestCase):
    """
    Negative binomial noise model unit tests that tests false positive and false negative rates.
    """

    def test_wald_de_nb(self):
        """

        :return:
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.INFO)

        self.noise_model = "nb"
        np.random.seed(1)
        self._test_wald_de_all_splines(ngenes=100, constrained=False)
        self._test_wald_de_all_splines(ngenes=100, constrained=True)
        return True


class TestContinuousDeNorm(_TestContinuousDe, unittest.TestCase):
    """
    Normal noise model unit tests that tests false positive and false negative rates.
    """

    def test_wald_de_norm(self):
        """

        :return:
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)

        self.noise_model = "norm"
        np.random.seed(1)
        self._test_wald_de_all_splines(ngenes=100, constrained=False)
        self._test_wald_de_all_splines(ngenes=100, constrained=True)
        return True


if __name__ == '__main__':
    unittest.main()
