import unittest
import logging
import numpy as np
import pandas as pd
import scipy.stats as stats

from batchglm.api.models.glm_nb import Simulator
import diffxpy.api as de


class TestEnrich(unittest.TestCase):

    def test_for_fatal(self):
        """
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)

        sim = Simulator(num_observations=50, num_features=10)
        sim.generate_sample_description(num_batches=0, num_conditions=2)
        sim.generate()

        test = de.test.wald(
            data=sim.X,
            factor_loc_totest="condition",
            formula_loc="~ 1 + condition",
            sample_description=sim.sample_description,
            gene_names=[str(x) for x in range(sim.X.shape[1])],
            training_strategy="DEFAULT",
            dtype="float64"
        )

        # Set up reference gene sets.
        rs = de.enrich.RefSets()
        rs.add(id="set1", source="manual", gene_ids=["1", "3"])
        rs.add(id="set2", source="manual", gene_ids=["5", "6"])

        enrich_test = de.enrich.test(
            ref=rs,
            det=test,
            de_threshold=0.05
        )
        return True


if __name__ == '__main__':
    unittest.main()
