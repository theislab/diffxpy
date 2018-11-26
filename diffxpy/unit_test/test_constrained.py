import unittest

import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.sparse
import anndata

from batchglm.api.models.nb_glm import Simulator, Estimator, InputData
import diffxpy.api as de


class TestSingle(unittest.TestCase):

    def test_null_distribution_wald_constrained(self, n_genes: int = 500):
        """
        Test if de.wald() with constraints generates a uniform p-value distribution
        if it is given data simulated based on the null model. Returns the p-value
        of the two-side Kolmgorov-Smirnov test for equality of the observed
        p-value distribution and a uniform distribution.

        n_cells is constant as the design matrix and constraints depend on it.

        :param n_genes: Number of genes to simulate (number of tests).
        """
        import batchglm.pkg_constants
        batchglm.pkg_constants.JACOBIAN_MODE = "tf"
        batchglm.pkg_constants.HESSIAN_MODE = "tf"

        n_cells = 2000

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

        dmat_est_loc = de.test.design_matrix(dmat=dmat_est)
        dmat_est_scale = de.test.design_matrix(dmat=dmat_est)

        # Build constraints:
        constraints_loc = np.zeros([2, dmat_est_loc.dims['design_params']])
        # Constraint 0: Account for perfect confounding at biological replicate and treatment level
        # by constraining biological replicate coefficients not to produce mean effects across conditions.
        constraints_loc[0, 3] = -1
        constraints_loc[0, 4:5] = 1
        # Constraint 1: Account for fact that first level of biological replicates was not absorbed into offset.
        constraints_loc[1, 1] = -1
        constraints_loc[1, 2:5] = 1

        constraints_scale = constraints_loc.copy()

        test = de.test.wald(
            data=sim.X,
            dmat_loc=dmat_est_loc.data_vars['design'],
            dmat_scale=dmat_est_scale.data_vars['design'],
            init_a="standard",
            init_b="standard",
            constraints_loc=constraints_loc,
            constraints_scale=constraints_scale,
            coef_to_test=["treatment1"],
            training_strategy="CONSTRAINED",
            dtype="float64"
        )
        summary = test.summary()

        # Compare p-value distribution under null model against uniform distribution.
        pval_h0 = stats.kstest(test.pval, 'uniform').pvalue

        print('KS-test pvalue for null model match of wald(): %f' % pval_h0)

        assert pval_h0 > 0.05, "KS-Test failed: pval_h0 is <= 0.05!"

        return pval_h0

    def test_null_distribution_wald_constrained_unequal_groups(self, n_genes: int = 200):
        """
        Test if de.wald() with constraints generates a uniform p-value distribution
        if it is given data simulated based on the null model. Returns the p-value
        of the two-side Kolmgorov-Smirnov test for equality of the observed
        p-value distribution and a uniform distribution.

        n_cells is constant as the design matrix and constraints depend on it.

        :param n_genes: Number of genes to simulate (number of tests).
        """
        import batchglm.pkg_constants
        batchglm.pkg_constants.JACOBIAN_MODE = "tf"
        batchglm.pkg_constants.HESSIAN_MODE = "tf"

        n_cells = 2000

        sim = Simulator(num_observations=n_cells, num_features=n_genes)
        sim.generate_sample_description(num_batches=0, num_conditions=0)
        sim.generate()

        # Build design matrix:
        dmat = np.zeros([n_cells, 6])
        dmat[:, 0] = 1
        dmat[:900, 1] = 1  # bio rep 1
        dmat[900:1000, 2] = 1  # bio rep 2
        dmat[1000:2000, 3] = 1  # bio rep 3
        dmat[2000:2500, 4] = 1  # bio rep 4
        dmat[1000:2500, 5] = 1  # condition effect
        coefficient_names = ['intercept', 'bio1', 'bio2', 'bio3', 'bio4', 'treatment1']
        dmat_est = pd.DataFrame(data=dmat, columns=coefficient_names)

        dmat_est_loc = de.test.design_matrix(dmat=dmat_est)
        dmat_est_scale = de.test.design_matrix(dmat=dmat_est)

        # Build constraints:
        constraints_loc = np.zeros([2, dmat_est_loc.dims['design_params']])
        # Constraint 0: Account for perfect confounding at biological replicate and treatment level
        # by constraining biological replicate coefficients not to produce mean effects across conditions.
        constraints_loc[0, 3] = -1
        constraints_loc[0, 4:5] = 1
        # Constraint 1: Account for fact that first level of biological replicates was not absorbed into offset.
        constraints_loc[1, 1] = -1
        constraints_loc[1, 2:5] = 1

        constraints_scale = constraints_loc.copy()

        test = de.test.wald(
            data=sim.X,
            dmat_loc=dmat_est_loc.data_vars['design'],
            dmat_scale=dmat_est_scale.data_vars['design'],
            init_a="standard",
            init_b="standard",
            constraints_loc=constraints_loc,
            constraints_scale=constraints_scale,
            coef_to_test=["treatment1"],
            training_strategy="CONSTRAINED",
            dtype="float64"
        )
        summary = test.summary()

        # Compare p-value distribution under null model against uniform distribution.
        pval_h0 = stats.kstest(test.pval, 'uniform').pvalue

        print('KS-test pvalue for null model match of wald(): %f' % pval_h0)

        assert pval_h0 > 0.05, "KS-Test failed: pval_h0 is <= 0.05!"

        return pval_h0

    def test_null_distribution_wald_constrained_2layer_confounding(self, n_genes: int = 200):
        """
        Test if de.wald() with constraints generates a uniform p-value distribution
        if it is given data simulated based on the null model. Returns the p-value
        of the two-side Kolmgorov-Smirnov test for equality of the observed
        p-value distribution and a uniform distribution.

        n_cells is constant as the design matrix and constraints depend on it.

        :param n_genes: Number of genes to simulate (number of tests).
        """
        import batchglm.pkg_constants
        batchglm.pkg_constants.JACOBIAN_MODE = "tf"
        batchglm.pkg_constants.HESSIAN_MODE = "tf"

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

        dmat_est_loc = de.test.design_matrix(dmat=dmat_est)
        dmat_est_scale = de.test.design_matrix(dmat=dmat_est.iloc[:, [0]])

        # Build constraints:
        constraints_loc = np.zeros([4, dmat_est_loc.dims['design_params']])
        # Constraints: bio reps
        constraints_loc[0, 6] = -1
        constraints_loc[0, 7:10] = 1
        constraints_loc[1, 2] = -1
        constraints_loc[1, 3:10] = 1
        # Constraints: tech reps
        constraints_loc[2, 12] = -1
        constraints_loc[2, 13:] = 1
        constraints_loc[3, 10] = -1
        constraints_loc[3, 11:] = 1

        constraints_scale = None

        test = de.test.wald(
            data=sim.X,
            dmat_loc=dmat_est_loc.data_vars['design'],
            dmat_scale=dmat_est_scale.data_vars['design'],
            init_a="standard",
            init_b="standard",
            constraints_loc=constraints_loc,
            constraints_scale=constraints_scale,
            coef_to_test=["treatment1"],
            training_strategy="CONSTRAINED",
            quick_scale=False,
            dtype="float64"
        )
        summary = test.summary()

        # Compare p-value distribution under null model against uniform distribution.
        pval_h0 = stats.kstest(test.pval, 'uniform').pvalue

        print('KS-test pvalue for null model match of wald(): %f' % pval_h0)

        assert pval_h0 > 0.05, "KS-Test failed: pval_h0 is <= 0.05!"

        return pval_h0

    def test_null_distribution_wald_constrained_multi(self, n_genes: int = 100):
        """
        Test if de.wald() for multiple coefficients with constraints
        generates a uniform p-value distribution
        if it is given data simulated based on the null model. Returns the p-value
        of the two-side Kolmgorov-Smirnov test for equality of the observed
        p-value distribution and a uniform distribution.

        n_cells is constant as the design matrix and constraints depend on it.

        :param n_genes: Number of genes to simulate (number of tests).
        """
        import batchglm.pkg_constants
        batchglm.pkg_constants.JACOBIAN_MODE = "tf"
        batchglm.pkg_constants.HESSIAN_MODE = "tf"

        n_cells = 3000

        sim = Simulator(num_observations=n_cells, num_features=n_genes)
        sim.generate_sample_description(num_batches=0, num_conditions=0)
        sim.generate()

        # Build design matrix:
        dmat = np.zeros([n_cells, 9])
        dmat[:, 0] = 1
        dmat[:500, 1] = 1  # bio rep 1
        dmat[500:1000, 2] = 1  # bio rep 2
        dmat[1000:1500, 3] = 1  # bio rep 3
        dmat[1500:2000, 4] = 1  # bio rep 4
        dmat[2000:2500, 5] = 1  # bio rep 5
        dmat[2500:3000, 6] = 1  # bio rep 6
        dmat[1000:2000, 7] = 1  # condition effect 1
        dmat[2000:3000, 8] = 1  # condition effect 2
        coefficient_names = ['intercept', 'bio1', 'bio2', 'bio3', 'bio4',
                             'bio5', 'bio6', 'treatment1', 'treatment2']
        dmat_est = pd.DataFrame(data=dmat, columns=coefficient_names)

        dmat_est_loc = de.test.design_matrix(dmat=dmat_est)
        dmat_est_scale = de.test.design_matrix(dmat=dmat_est)

        # Build constraints:
        constraints_loc = np.zeros([3, dmat_est_loc.dims['design_params']])
        constraints_loc[0, 3] = -1
        constraints_loc[0, 4] = 1

        constraints_loc[1, 5] = -1
        constraints_loc[1, 6] = 1

        constraints_loc[2, 1] = -1
        constraints_loc[2, 2:7] = 1

        constraints_scale = constraints_loc.copy()

        test = de.test.wald(
            data=sim.X,
            dmat_loc=dmat_est_loc.data_vars['design'],
            dmat_scale=dmat_est_scale.data_vars['design'],
            constraints_loc=constraints_loc,
            constraints_scale=constraints_scale,
            coef_to_test=["treatment1", "treatment2"],
            training_strategy="CONSTRAINED",
            dtype="float64"
        )
        summary = test.summary()

        # Compare p-value distribution under null model against uniform distribution.
        pval_h0 = stats.kstest(test.pval, 'uniform').pvalue

        print('KS-test pvalue for null model match of wald(): %f' % pval_h0)

        assert pval_h0 > 0.05, "KS-Test failed: pval_h0 is <= 0.05!"

        return pval_h0


if __name__ == '__main__':
    unittest.main()
