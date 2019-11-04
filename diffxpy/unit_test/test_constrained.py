import logging
import unittest

import numpy as np
import pandas as pd
import scipy.stats as stats

from batchglm.api.models.numpy.glm_nb import Simulator
import diffxpy.api as de


class TestConstrained(unittest.TestCase):

    def test_forfatal_from_string(self):
        """
        Test if _from_string interface is working.

        n_cells is constant as the design matrix and constraints depend on it.
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)

        np.random.seed(1)
        n_cells = 2000
        n_genes = 2

        sim = Simulator(num_observations=n_cells, num_features=n_genes)
        sim.generate_sample_description(num_batches=0, num_conditions=0)
        sim.generate()

        # Build design matrix:
        dmat = np.zeros([n_cells, 6])
        dmat[:, 0] = 1
        dmat[:500, 1] = 1  # bio rep 1
        dmat[500:1000, 2] = 1  # bio rep 2
        dmat[1000:1500, 3] = 1  # bio rep 3
        dmat[1500:2000, 4] = 1  # bio rep 4
        dmat[1000:2000, 5] = 1  # condition effect
        coefficient_names = ['intercept', 'bio1', 'bio2', 'bio3', 'bio4', 'treatment1']
        dmat_est = pd.DataFrame(data=dmat, columns=coefficient_names)

        dmat_est_loc, _ = de.utils.design_matrix(dmat=dmat_est, return_type="dataframe")
        dmat_est_scale, _ = de.utils.design_matrix(dmat=dmat_est, return_type="dataframe")

        # Build constraints:
        constraints_loc = de.utils.constraint_matrix_from_string(
            dmat=dmat_est_loc.values,
            coef_names=dmat_est_loc.columns,
            constraints=["bio1+bio2=0", "bio3+bio4=0"]
        )
        constraints_scale = de.utils.constraint_matrix_from_string(
            dmat=dmat_est_scale.values,
            coef_names=dmat_est_scale.columns,
            constraints=["bio1+bio2=0", "bio3+bio4=0"]
        )

        test = de.test.wald(
            data=sim.input_data,
            dmat_loc=dmat_est_loc,
            dmat_scale=dmat_est_scale,
            constraints_loc=constraints_loc,
            constraints_scale=constraints_scale,
            coef_to_test=["treatment1"]
        )
        _ = test.summary()

    def test_forfatal_from_dict(self):
        """
        Test if dictionary-based constraint interface is working.
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)

        np.random.seed(1)
        n_cells = 2000
        n_genes = 2

        sim = Simulator(num_observations=n_cells, num_features=n_genes)
        sim.generate_sample_description(num_batches=0, num_conditions=0)
        sim.generate()

        # Build design matrix:
        sample_description = pd.DataFrame({
            "cond": ["cond"+str(i // 1000) for i in range(n_cells)],
            "batch": ["batch"+str(i // 500) for i in range(n_cells)]
        })

        test = de.test.wald(
            data=sim.input_data,
            sample_description=sample_description,
            formula_loc="~1+cond+batch",
            formula_scale="~1+cond+batch",
            constraints_loc={"batch": "cond"},
            constraints_scale={"batch": "cond"},
            coef_to_test=["cond[T.cond1]"]
        )
        _ = test.summary()

    def test_null_distribution_wald_constrained(self, n_genes: int = 100):
        """
        Test if de.wald() with constraints generates a uniform p-value distribution
        if it is given data simulated based on the null model. Returns the p-value
        of the two-side Kolmgorov-Smirnov test for equality of the observed
        p-value distribution and a uniform distribution.

        n_cells is constant as the design matrix and constraints depend on it.

        :param n_genes: Number of genes to simulate (number of tests).
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)

        np.random.seed(1)
        n_cells = 2000
        sim = Simulator(num_observations=n_cells, num_features=n_genes)
        sim.generate_sample_description(num_batches=0, num_conditions=0)
        sim.generate()

        # Build design matrix:
        sample_description = pd.DataFrame({
            "cond": ["cond" + str(i // 1000) for i in range(n_cells)],
            "batch": ["batch" + str(i // 500) for i in range(n_cells)]
        })

        test = de.test.wald(
            data=sim.input_data,
            sample_description=sample_description,
            formula_loc="~1+cond+batch",
            formula_scale="~1+cond+batch",
            constraints_loc={"batch": "cond"},
            constraints_scale={"batch": "cond"},
            coef_to_test=["cond[T.cond1]"]
        )
        _ = test.summary()

        # Compare p-value distribution under null model against uniform distribution.
        pval_h0 = stats.kstest(test.pval, 'uniform').pvalue

        logging.getLogger("diffxpy").info('KS-test pvalue for null model match of wald(): %f' % pval_h0)
        assert pval_h0 > 0.05, "KS-Test failed: pval_h0 is <= 0.05!"

        return True

    def _test_null_distribution_wald_constrained_2layer(self, n_genes: int = 100):
        """
        Test if de.wald() with constraints generates a uniform p-value distribution
        if it is given data simulated based on the null model. Returns the p-value
        of the two-side Kolmgorov-Smirnov test for equality of the observed
        p-value distribution and a uniform distribution.

        n_cells is constant as the design matrix and constraints depend on it.

        :param n_genes: Number of genes to simulate (number of tests).
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)

        np.random.seed(1)
        n_cells = 12000
        sim = Simulator(num_observations=n_cells, num_features=n_genes)
        sim.generate_sample_description(num_batches=0, num_conditions=0)
        sim.generate()

        # Build design matrix:
        dmat = np.zeros([n_cells, 14])
        dmat[:, 0] = 1
        dmat[6000:12000, 1] = 1  # condition effect
        dmat[:1000, 2] = 1  # bio rep 1 - treated 1
        dmat[1000:3000, 3] = 1  # bio rep 2 - treated 2
        dmat[3000:5000, 4] = 1  # bio rep 3 - treated 3
        dmat[5000:6000, 5] = 1  # bio rep 4 - treated 4
        dmat[6000:7000, 6] = 1  # bio rep 5 - untreated 1
        dmat[7000:9000, 7] = 1  # bio rep 6 - untreated 2
        dmat[9000:11000, 8] = 1  # bio rep 7 - untreated 3
        dmat[11000:12000, 9] = 1  # bio rep 8 - untreated 4
        dmat[1000:2000, 10] = 1  # tech rep 1
        dmat[7000:8000, 10] = 1  # tech rep 1
        dmat[2000:3000, 11] = 1  # tech rep 2
        dmat[8000:9000, 11] = 1  # tech rep 2
        dmat[3000:4000, 12] = 1  # tech rep 3
        dmat[9000:10000, 12] = 1  # tech rep 3
        dmat[4000:5000, 13] = 1  # tech rep 4
        dmat[10000:11000, 13] = 1  # tech rep 4

        coefficient_names = ['intercept', 'treatment1',
                             'bio1', 'bio2', 'bio3', 'bio4', 'bio5', 'bio6', 'bio7', 'bio8',
                             'tech1', 'tech2', 'tech3', 'tech4']
        dmat_est = pd.DataFrame(data=dmat, columns=coefficient_names)

        dmat_est_loc = de.utils.design_matrix(dmat=dmat_est, return_type="dataframe")
        dmat_est_scale = de.utils.design_matrix(dmat=dmat_est.iloc[:, [0]], return_type="dataframe")

        # Build constraints:
        constraints_loc = de.utils.constraint_matrix_from_string(
            dmat=dmat_est_loc.values,
            coef_names=dmat_est_loc.columns,
            constraints=["bio1+bio2=0",
                         "bio3+bio4=0",
                         "bio5+bio6=0",
                         "bio7+bio8=0",
                         "tech1+tech2=0",
                         "tech3+tech4=0"]
        )
        constraints_scale = None

        test = de.test.wald(
            data=sim.input_data,
            dmat_loc=dmat_est_loc,
            dmat_scale=dmat_est_scale,
            constraints_loc=constraints_loc,
            constraints_scale=constraints_scale,
            coef_to_test=["treatment1"]
        )
        _ = test.summary()

        # Compare p-value distribution under null model against uniform distribution.
        pval_h0 = stats.kstest(test.pval, 'uniform').pvalue

        logging.getLogger("diffxpy").info('KS-test pvalue for null model match of wald(): %f' % pval_h0)
        assert pval_h0 > 0.05, "KS-Test failed: pval_h0 is <= 0.05!"

        return True


if __name__ == '__main__':
    unittest.main()
