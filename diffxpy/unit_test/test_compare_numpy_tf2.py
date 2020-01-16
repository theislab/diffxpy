import unittest
import logging
import diffxpy.api as de
import numpy as np


class TestBackendsNb(unittest.TestCase):

    """
    Negative binomial noise model unit tests that test whether the wald test results
    in the same logfoldchange, p- and q-values and the coefficents are the same after
    fitting when using the same simulated data.
    """

    def __init__(self, *args, **kwargs):

        super(TestBackendsNb, self).__init__(*args, **kwargs)

        from batchglm.api.models.numpy.glm_nb import Simulator
        self.sim = Simulator(num_observations=10000, num_features=200)
        self.sim.generate_sample_description(num_batches=0, num_conditions=4)
        self.sim.generate_params()
        self.sim.generate_data()

        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)

        self.numpy_results = de.test.wald(
            data=self.sim.input_data,
            sample_description=self.sim.sample_description,
            factor_loc_totest="condition",
            formula_loc="~ 1 + condition + batch",
            noise_model='nb',
            backend='numpy'
        )
        _ = self.numpy_results.summary()

        self.tf2_results = de.test.wald(
            data=self.sim.input_data,
            sample_description=self.sim.sample_description,
            factor_loc_totest="condition",
            formula_loc="~ 1 + condition + batch",
            noise_model='nb',
            backend='tf2'
        )
        _ = self.tf2_results.summary()

    """
    Test with numpy:
    """

    def test_coeff_similarity(self):

        a_var_max_diff = np.max(np.abs(self.tf2_results.model_estim.a_var-self.numpy_results.model_estim.a_var))
        b_var_max_diff = np.max(np.abs(self.tf2_results.model_estim.b_var-self.numpy_results.model_estim.b_var))
        assert a_var_max_diff < 1e-6 and b_var_max_diff < 1e-6, \
            ("a_var_max_diff: %f, b_var_max_diff: %f", a_var_max_diff, b_var_max_diff)

        return True

    def test_logfoldchange_similarity(self):
        max_diff = np.max(np.abs(self.tf2_results.summary()['log2fc'].values-self.numpy_results.summary()['log2fc'].values))
        assert max_diff < 1e-12, ("log_fold_change difference: %f > 1e-12", max_diff)

        return True

    def test_pval_similarity(self):
        max_diff = np.max(np.abs(self.tf2_results.pval-self.numpy_results.pval))
        assert max_diff < 1e-12, ("p-val difference: %f > 1e-12", max_diff)

        return True

    def test_qval_similarity(self):
        max_diff = np.max(np.abs(self.tf2_results.summary()['qval'].values-self.numpy_results.summary()['qval'].values))
        assert max_diff < 1e-12, ("q-val difference: %f > 1e-12", max_diff)

        return True

if __name__ == '__main__':
    unittest.main()
