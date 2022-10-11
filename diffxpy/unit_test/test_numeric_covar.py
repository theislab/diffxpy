import unittest

import numpy as np
import logging

from batchglm.models.glm_nb import Model as NBModel
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

        model = NBModel()
        model.generate_artificial_data(
            n_obs=2000,
            n_vars=2,
            num_batches=0,
            num_conditions=2,
        )

        sample_description = model.sample_description
        sample_description["numeric1"] = np.random.random(size=2000)
        sample_description["numeric2"] = np.random.random(size=2000)

        test = de.test.wald(
            data=model.x,
            gene_names=model.features,
            sample_description=sample_description,
            formula_loc="~ 1 + condition + numeric1 + numeric2",
            formula_scale="~ 1",
            factor_loc_totest="condition",
            as_numeric=["numeric1", "numeric2"],
            training_strategy="DEFAULT"
        )
        # Check that number of coefficients is correct.
        assert test.model_estim.model_container.theta_location.shape[0] == 4

        return True


if __name__ == '__main__':
    unittest.main()
