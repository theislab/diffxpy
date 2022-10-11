import unittest
import logging
import numpy as np
import pandas as pd

from batchglm.models.glm_nb import Model as NBModel
from batchglm.models.glm_norm import Model as NormModel

import diffxpy.api as de


class _TestSingleFullRank(unittest.TestCase):

    def _test_single_full_rank(self):
        """
        Test if de.wald() generates a uniform p-value distribution
        if it is given data simulated based on the null model. Returns the p-value
        of the two-side Kolmgorov-Smirnov test for equality of the observed 
        p-value distribution and a uniform distribution.

        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        :param noise_model: Noise model to use for data fitting.
        """
        if self.noise_model == "nb":
            model = NBModel()
            rand_fn_scale = lambda shape: np.random.uniform(1, 2, shape)
        elif self.noise_model == "norm":
            model = NormModel()
            rand_fn_scale = lambda shape: np.random.uniform(1, 2, shape)
        else:
            raise ValueError("noise model %s not recognized" % self.noise_model)

        model.generate_artificial_data(
            n_obs=200,
            n_vars=2,
            num_batches=0,
            num_conditions=0,
            rand_fn_scale=rand_fn_scale
        )

        random_sample_description = pd.DataFrame({
            "condition": [str(x) for x in np.random.randint(2, size=200)]
        })

        try:
            random_sample_description["batch"] = random_sample_description["condition"]
            _ = de.test.wald(
                data=model.x,
                gene_names=model.features,
                sample_description=random_sample_description,
                factor_loc_totest="condition[T.1]",
                formula_loc="~ 1 + condition + batch",
                noise_model=self.noise_model
            )
        except ValueError as error:
            logging.getLogger("diffxpy").info(error)
        else:
            raise ValueError("rank error was erroneously not thrown on under-determined unconstrained system")

        try:
            random_sample_description["batch"] = [
                x + str(np.random.randint(0, 2)) for x in random_sample_description["condition"].values
            ]
            _ = de.test.wald(
                data=model.x,
                gene_names=model.features,
                sample_description=random_sample_description,
                factor_loc_totest="condition[T.1]",
                formula_loc="~ 1 + condition + batch",
                constraints_loc={"batch": "condition"},
                noise_model=self.noise_model
            )
        except ValueError as error:
            raise ValueError("rank error was erroneously thrown on defined constrained system")

    def test_single_full_rank(
            self):
        """
        Test if error is thrown if input is not full rank.
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.INFO)

        np.random.seed(1)
        self.noise_model = "nb"
        return self._test_single_full_rank()


if __name__ == '__main__':
    unittest.main()
