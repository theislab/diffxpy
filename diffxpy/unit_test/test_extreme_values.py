import unittest
import logging

import numpy as np
import pandas as pd

from batchglm.models.glm_nb import Model as NBModel
import diffxpy.api as de


class TestExtremeValues(unittest.TestCase):

    def test_t_test_zero_variance(self):
        """
        Test if T-test works if it is given genes with zero variance.
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)

        np.random.seed(1)
        model = NBModel()
        model.generate_artificial_data(
            n_obs=1000,
            n_vars=10,
            num_batches=0,
            num_conditions=0,
        )
        model.x[:, 0] = 0
        model.x[:, 1] = 5

        random_sample_description = pd.DataFrame({
            "condition": np.random.randint(2, size=1000)
        })

        test = de.test.t_test(
            data=model.x,
            gene_names=model.features,
            sample_description=random_sample_description,
            grouping="condition",
            is_sig_zerovar=True
        )

        assert np.isnan(test.pval[0]) and test.pval[1] == 1, \
            "t test did not assign p-value of zero to groups with zero variance and same mean, %f, %f" % \
            (test.pval[0], test.pval[1])
        return True

    def test_rank_test_zero_variance(self):
        """
        Test if rank test works if it is given genes with zero variance.
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)

        np.random.seed(1)
        model = NBModel()
        model.generate_artificial_data(
            n_obs=1000,
            n_vars=10,
            num_batches=0,
            num_conditions=0,
        )
        model.x[:, 0] = 0
        model.x[:, 1] = 5

        random_sample_description = pd.DataFrame({
            "condition": np.random.randint(2, size=1000)
        })

        test = de.test.rank_test(
            data=model.x,
            gene_names=model.features,
            sample_description=random_sample_description,
            grouping="condition",
            is_sig_zerovar=True
        )

        assert np.isnan(test.pval[0]) and test.pval[1] == 1, \
            "rank test did not assign p-value of zero to groups with zero variance and same mean, %f, %f" % \
            (test.pval[0], test.pval[1])
        return True


if __name__ == '__main__':
    unittest.main()
