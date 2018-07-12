import abc
import logging

import numpy as np

try:
    import xarray as xr
except ImportError:
    xr = None

try:
    import anndata
except ImportError:
    anndata = None

import data as data_utils
from . import stats


class _Estimation(metaclass=abc.ABCMeta):
    """
    Dummy class specifying all needed methods / parameters necessary for DifferentialExpressionTest.
    Useful for type hinting.
    """

    @property
    @abc.abstractmethod
    def design(self) -> np.ndarray:
        pass

    @abc.abstractmethod
    def probs(self) -> np.ndarray:
        pass

    @abc.abstractmethod
    def log_probs(self) -> np.ndarray:
        pass

    @property
    @abc.abstractmethod
    def loss(self, **kwargs) -> np.ndarray:
        pass

    @property
    @abc.abstractmethod
    def gradient(self, **kwargs) -> np.ndarray:
        pass

    @property
    @abc.abstractmethod
    def hessian_diagonal(self, **kwargs) -> np.ndarray:
        pass


class DifferentialExpressionTest:
    full_estim: _Estimation
    reduced_estim: _Estimation

    def __init__(self, full_estim: _Estimation, reduced_estim: _Estimation):
        self.full_estim = full_estim
        self.reduced_estim = reduced_estim

    def reduced_model_gradient(self):
        return self.reduced_estim.gradient

    def full_model_gradient(self):
        return self.full_estim.gradient

    def likelihood_ratio_test(self):
        full = np.sum(self.full_estim.log_probs(), axis=0)
        reduced = np.sum(self.reduced_estim.log_probs(), axis=0)

        pval = stats.likelihood_ratio_test(
            ll_full=full,
            ll_reduced=reduced,
            df=self.full_estim.design.shape[-1] - self.reduced_estim.design.shape[-1]
        )

        return pval


def test(data, full_formula, reduced_formula, sample_description=None, noise_model="nb", close_sessions=True):
    full_design = None
    reduced_design = None
    if anndata is not None and isinstance(data, anndata.AnnData):
        if sample_description is None:
            full_design = data_utils.design_matrix_from_anndata(
                dataset=data,
                formula=full_formula
            )
            reduced_design = data_utils.design_matrix_from_anndata(
                dataset=data,
                formula=full_design.design_info.subset(reduced_formula)
            )

        # X = data.X
    elif xr is not None and isinstance(data, xr.Dataset):
        if sample_description is None:
            full_design = data_utils.design_matrix_from_xarray(
                dataset=data,
                dim="observations",
                formula=full_formula
            )
            reduced_design = data_utils.design_matrix_from_xarray(
                dataset=data,
                dim="observations",
                formula=full_design.design_info.subset(reduced_formula)
            )

        # X = data["X"]
    else:
        if sample_description is None:
            raise ValueError(
                "Please specify `sample_description` or provide `data` as xarray.Dataset or anndata.AnnData " +
                "with corresponding sample annotations"
            )

        # X = data

    # in each case: if `sample_description` is specified, overwrite previous designs
    if sample_description is not None:
        full_design = data_utils.design_matrix(
            sample_description=sample_description,
            formula=full_formula
        )
        reduced_design = data_utils.design_matrix(
            sample_description=sample_description,
            formula=full_design.design_info.subset(reduced_formula)
        )

    logger = logging.getLogger(__name__)

    if noise_model == "nb" or noise_model == "negative_binomial":
        import api.models.nb_glm as test_model

        logger.info("Estimating reduced model...")
        reduced_input_data = test_model.InputData(data=data, design=reduced_design)
        estim = test_model.Estimator(input_data=reduced_input_data)
        estim.initialize()
        estim.train(learning_rate=0.5, loss_history_size=200, stop_at_loss_change=0.05)
        estim.train(learning_rate=0.05, loss_history_size=200, stop_at_loss_change=0.05)
        if close_sessions:
            reduced_model = estim.finalize()
        else:
            reduced_model = estim
        logger.info("Estimation of reduced model ready")

        # find columns of `full_model` which are identical to `reduced_model`
        indices = np.arange(len(full_design.design_info.column_name_indexes))
        indices = np.unique(np.concatenate([
            indices[v] for i, v in full_design.design_info.term_name_slices.items() if
            i in reduced_design.design_info.term_names
        ]))

        # initialize `a` and `b` of the full model with estimated values from the reduced model
        shape = list(reduced_model.a.shape)
        shape[-2] = full_design.shape[-1]

        a = np.tile(
            np.sqrt(np.nextafter(0, 1, dtype=reduced_model.a.dtype)),
            reps=shape
        )
        b = a.copy()

        a[indices] = reduced_model.a
        a_intercept = a[[0]]
        a_slopes = a[1:]

        b[indices] = reduced_model.b
        b_intercept = b[[0]]
        b_slopes = b[1:]

        logger.info("Estimating full model...")
        full_input_data = test_model.InputData(data=data, design=reduced_design)
        estim = test_model.Estimator(
            input_data=full_input_data,
            init_a_intercept=a_intercept,
            init_a_slopes=a_slopes,
            init_b_intercept=b_intercept,
            init_b_slopes=b_slopes,
        )
        estim.initialize()
        estim.train(learning_rate=0.5, loss_history_size=200, stop_at_loss_change=0.05)
        estim.train(learning_rate=0.05, loss_history_size=200, stop_at_loss_change=0.05)
        if close_sessions:
            full_model = estim.finalize()
        else:
            full_model = estim
        logger.info("Estimation of full model ready")

        de_test = DifferentialExpressionTest(full_model, reduced_model)
        return de_test


def two_sample(cond: str):  # wrapper for test with '~ 1 + cond'
    pass

def many2many():
    pass