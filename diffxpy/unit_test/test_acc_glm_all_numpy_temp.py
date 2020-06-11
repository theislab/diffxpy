import logging
import anndata
import numpy as np
import scipy.sparse
import unittest

import batchglm.api as glm
import diffxpy.api as de

glm.setup_logging(verbosity="WARNING", stream="STDOUT")
logger = logging.getLogger(__name__)


class TestAccuracyGlmNb(
    unittest.TestCase
):
    """
    Test whether optimizers yield exact results for negative binomial distributed data.
    """

    def test_full_nb(self):
        logging.getLogger("batchglm").setLevel(logging.INFO)
        logger.error("TestAccuracyGlmNb.test_full_nb()")

        np.random.seed(1)
        adata = anndata.read_h5ad("/Users/david.fischer/Desktop/test.h5ad")
        TF = "Ascl1"
        temp = de.test.continuous_1d(
            data=adata[:, :10],
            formula_loc="~ 1 +" + TF + "_log",  # + " + log_sf",
            formula_scale="~ 1 +" + TF + "_log",  # + " + log_sf",
            factor_loc_totest=TF + "_log",
            continuous=TF + "_log",
            as_numeric=[TF + "_log"],  # "log_sf"],
            df=4,
            quick_scale=False,
            init_a="all_zero",
            size_factors=None,
            noise_model="poisson",
            backend="numpy"
        )
        _ = temp.summary()


if __name__ == '__main__':
    unittest.main()
