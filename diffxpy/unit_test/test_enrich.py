import unittest
import logging

from batchglm.models.glm_nb import Model as NBModel
import diffxpy.api as de


class TestEnrich(unittest.TestCase):

    def test_for_fatal(self):
        """
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)

        model = NBModel()
        model.generate_artificial_data(
            n_obs=50,
            n_vars=10,
            num_batches=0,
            num_conditions=2
        )

        test = de.test.wald(
            data=model.x,
            gene_names=[str(x) for x in range(model.x.shape[1])],
            factor_loc_totest="condition",
            formula_loc="~ 1 + condition",
            sample_description=model.sample_description,
            training_strategy="DEFAULT",
            dtype="float64"
        )

        # Set up reference gene sets.
        rs = de.enrich.RefSets()
        rs.add(id="set1", source="manual", gene_ids=["1", "3"])
        rs.add(id="set2", source="manual", gene_ids=["5", "6"])

        for i in [True, False]:
            for j in [True, False]:
                enrich_test_i = de.enrich.test(
                    ref=rs,
                    det=test,
                    threshold=0.05,
                    incl_all_zero=i,
                    clean_ref=j,
                )
                _ = enrich_test_i.summary()
                _ = enrich_test_i.significant_set_ids()
                _ = enrich_test_i.significant_sets()
                _ = enrich_test_i.set_summary(id="set1")

        return True


if __name__ == '__main__':
    unittest.main()
