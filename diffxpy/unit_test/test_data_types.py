import unittest
import logging

import numpy as np
import pandas as pd
import scipy.sparse
import anndata

from batchglm.api.models.numpy.glm_nb import Simulator
import diffxpy.api as de


class TestDataTypesSingle(unittest.TestCase):

    def _test_wald(self, data, sample_description, gene_names=None):
        test = de.test.wald(
            data=data,
            sample_description=sample_description,
            gene_names=gene_names,
            factor_loc_totest="condition",
            formula_loc="~ 1 + condition",
            noise_model="nb",
            batch_size=5
        )
        _ = test.summary()

    def _test_lrt(self, data, sample_description, gene_names=None):
        test = de.test.lrt(
            data=data,
            sample_description=sample_description,
            gene_names=gene_names,
            full_formula_loc="~ 1 + condition",
            reduced_formula_loc="~ 1",
            noise_model="nb"
        )
        _ = test.summary()

    def _test_t_test(self, data, sample_description, gene_names=None):
        test = de.test.t_test(
            data=data,
            sample_description=sample_description,
            gene_names=gene_names,
            grouping="condition"
        )
        _ = test.summary()

    def _test_rank(self, data, sample_description, gene_names=None):
        test = de.test.rank_test(
            data=data,
            sample_description=sample_description,
            gene_names=gene_names,
            grouping="condition"
        )
        _ = test.summary()

    def simulate(self, n_cells: int = 200, n_genes: int = 2):
        sim = Simulator(num_observations=n_cells, num_features=n_genes)
        sim.generate_sample_description(num_batches=0, num_conditions=0)
        sim.generate()

        random_sample_description = pd.DataFrame({
            "condition": np.random.randint(2, size=sim.input_data.num_observations)
        })
        return sim.x, random_sample_description

    def _test_numpy(self, sparse):
        data, sample_description = self.simulate()
        gene_names = ["gene" + str(i) for i in range(data.shape[1])]
        if sparse:
            data = scipy.sparse.csr_matrix(data)

        self._test_wald(data=data, sample_description=sample_description, gene_names=gene_names)
        self._test_lrt(data=data, sample_description=sample_description, gene_names=gene_names)
        self._test_t_test(data=data, sample_description=sample_description, gene_names=gene_names)
        self._test_rank(data=data, sample_description=sample_description, gene_names=gene_names)

    def _test_anndata(self, sparse):
        data, sample_description = self.simulate()
        gene_names = ["gene" + str(i) for i in range(data.shape[1])]
        if sparse:
            data = scipy.sparse.csr_matrix(data)

        data = anndata.AnnData(data)
        data.var_names = gene_names
        self._test_wald(data=data, sample_description=sample_description)
        self._test_lrt(data=data, sample_description=sample_description)
        self._test_t_test(data=data, sample_description=sample_description)
        self._test_rank(data=data, sample_description=sample_description)

    def _test_anndata_raw(self, sparse):
        data, sample_description = self.simulate()
        gene_names = ["gene" + str(i) for i in range(data.shape[1])]
        if sparse:
            data = scipy.sparse.csr_matrix(data)

        data = anndata.AnnData(data)
        data.var_names = gene_names
        data.raw = data
        self._test_wald(data=data.raw, sample_description=sample_description)
        self._test_lrt(data=data.raw, sample_description=sample_description)
        self._test_t_test(data=data, sample_description=sample_description)
        self._test_rank(data=data, sample_description=sample_description)

    def test_numpy(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)

        self._test_numpy(sparse=False)
        self._test_numpy(sparse=True)

        return True

    def test_anndata(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("diffxpy").setLevel(logging.WARNING)

        self._test_anndata(sparse=False)
        self._test_anndata(sparse=True)
        self._test_anndata_raw(sparse=False)
        self._test_anndata_raw(sparse=True)

        return True


if __name__ == '__main__':
    unittest.main()
