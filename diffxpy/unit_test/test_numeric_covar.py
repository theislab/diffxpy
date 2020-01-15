import unittest

import numpy as np
import logging

from batchglm.api.models.numpy.glm_nb import Simulator
import diffxpy.api as de


class TestNumericCovar(unittest.TestCase):

    def test(self):
        """
        Check that factors that are numeric receive the correct number of coefficients.

        :return:
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)

        sim = Simulator(num_observations=2000, num_features=2)
        sim.generate_sample_description(num_batches=0, num_conditions=2)
        sim.generate_params()
        sim.generate_data()

        sample_description = sim.sample_description
        sample_description["numeric1"] = np.random.random(size=sim.nobs)
        sample_description["numeric2"] = np.random.random(size=sim.nobs)

        test = de.test.wald(
            data=sim.input_data,
            sample_description=sample_description,
            formula_loc="~ 1 + condition + numeric1 + numeric2",
            formula_scale="~ 1",
            factor_loc_totest="condition",
            as_numeric=["numeric1", "numeric2"],
            training_strategy="DEFAULT"
        )
        # Check that number of coefficients is correct.
        assert test.model_estim.a_var.shape[0] == 4

        return True


if __name__ == '__main__':
    unittest.main()
