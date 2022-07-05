import logging
import numpy as np
import unittest
import scanpy as sc
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
        adata = sc.datasets.pbmc3k()
        tf = "MALAT1"
        ind = adata.var.index.get_loc(tf)
        log_cd4 = sc.pp.log1p(adata[:, tf].X.todense())
        adata.obs[tf + "_log"] = log_cd4
        temp = de.test.continuous_1d(
            data=adata[:, (ind - 5):(ind + 5)],
            formula_loc="~ 1 +" + tf + "_log",  # + " + log_sf",
            formula_scale="~ 1",
            factor_loc_totest=tf + "_log",
            continuous=tf + "_log",
            as_numeric=[tf + "_log"],  # "log_sf"],
            df=4,
            quick_scale=False,
            init_a="all_zero",
            size_factors=None,
            noise_model="nb",
            backend="numpy"
        )
        _ = temp.summary()


if __name__ == '__main__':
    unittest.main()
