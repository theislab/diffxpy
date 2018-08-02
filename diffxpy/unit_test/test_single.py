import unittest

import numpy as np
import pandas as pd
import scipy.stats as stats

from batchglm.api.models.nb_glm import Simulator, Estimator, InputData
import diffxpy as de


class TestSingle(unittest.TestCase):
    
    def test_wald(self, n_cells: int = 1000, n_genes: int = 1000):
        """
        Test if de.test_wald_loc() generates a uniform p-value distribution
        if it is given data simulated based on the null model. Returns the p-value
        of the two-side Kolmgorov-Smirnov test for equality of the observed 
        p-value distriubution and a uniform distribution.

        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        """
        
        sim = Simulator(num_observations=n_cells, num_features=n_genes)
        sim.generate_sample_description(num_batches=0, num_confounders=0)
        sim.generate()
        
        random_sample_description = pd.DataFrame({
            "condition": np.random.randint(2, size=sim.num_observations)
        })
        
        test = de.test_wald_loc(
            data=sim.X,
            factor_loc_totest="condition",
            formula="~ 1 + condition",
            sample_description=random_sample_description,
            training_strategy='QUICK'
        )
        
        # Compare p-value distribution under null model against uniform distribution.
        pval_h0 = stats.kstest(test.pval, 'uniform').pvalue
        
        print('KS-test pvalue for null model match of test_wald_loc(): %f' % pval_h0)
        return pval_h0
    
    def test_lrt(self, n_cells: int = 1000, n_genes: int = 1000):
        """
        Test if de.test_lrt() generates a uniform p-value distribution
        if it is given data simulated based on the null model. Returns the p-value
        of the two-side Kolmgorov-Smirnov test for equality of the observed 
        p-value distriubution and a uniform distribution.

        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        """
        
        sim = Simulator(num_observations=n_cells, num_features=n_genes)
        sim.generate_sample_description(num_batches=0, num_confounders=0)
        sim.generate()
        
        random_sample_description = pd.DataFrame({
            "condition": np.random.randint(2, size=sim.num_observations)
        })
        
        test = de.test_lrt(
            data=sim.X,
            full_formula="~ 1 + condition",
            reduced_formula="~ 1",
            sample_description=random_sample_description,
            training_strategy='QUICK'
        )
        
        # Compare p-value distribution under null model against uniform distribution.
        pval_h0 = stats.kstest(test.pval, 'uniform').pvalue
        
        print('KS-test pvalue for null model match of test_lrt(): %f' % pval_h0)
        return pval_h0
    
    def test_t_test(self, n_cells: int = 1000, n_genes: int = 1000):
        """
        Test if de.test_t_test() generates a uniform p-value distribution
        if it is given data simulated based on the null model. Returns the p-value
        of the two-side Kolmgorov-Smirnov test for equality of the observed 
        p-value distriubution and a uniform distribution.

        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        """
        
        sim = Simulator(num_observations=n_cells, num_features=n_genes)
        sim.generate_sample_description(num_batches=0, num_confounders=0)
        sim.generate()
        
        random_sample_description = pd.DataFrame({
            "condition": np.random.randint(2, size=sim.num_observations)
        })
        
        test = de.test_t_test(
            data=sim.X,
            grouping="condition",
            sample_description=random_sample_description
        )
        
        # Compare p-value distribution under null model against uniform distribution.
        pval_h0 = stats.kstest(test.pval, 'uniform').pvalue
        
        print('KS-test pvalue for null model match of test_t_test(): %f' % pval_h0)
        return pval_h0
    
    def test_wilcoxon(self, n_cells: int = 1000, n_genes: int = 1000):
        """
        Test if de.test_wilcoxon() generates a uniform p-value distribution
        if it is given data simulated based on the null model. Returns the p-value
        of the two-side Kolmgorov-Smirnov test for equality of the observed 
        p-value distriubution and a uniform distribution.

        :param n_cells: Number of cells to simulate (number of observations per test).
        :param n_genes: Number of genes to simulate (number of tests).
        """
        
        sim = Simulator(num_observations=n_cells, num_features=n_genes)
        sim.generate_sample_description(num_batches=0, num_confounders=0)
        sim.generate()
        
        random_sample_description = pd.DataFrame({
            "condition": np.random.randint(2, size=sim.num_observations)
        })
        
        test = de.test_wilcoxon(
            data=sim.X,
            grouping="condition",
            sample_description=random_sample_description
        )
        
        # Compare p-value distribution under null model against uniform distribution.
        pval_h0 = stats.kstest(test.pval, 'uniform').pvalue
        
        print('KS-test pvalue for null model match of test_wilcoxon(): %f' % pval_h0)
        return pval_h0


if __name__ == '__main__':
    unittest.main()
