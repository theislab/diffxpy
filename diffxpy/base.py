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
from . import stattest


class DifferentialExpressionTest:

    def __init__(self, X, full_model, reduced_model):
        self.X = X
        self.full_model = full_model
        self.reduced_model = reduced_model

    def likelihood_ratio_test(self):
        full = np.sum(self.full_model.log_probs(self.X), axis=0)
        reduced = np.sum(self.reduced_model.log_probs(self.X), axis=0)

        pval = stattest.likelihood_ratio_test(
            ll_full=full,
            ll_reduced=reduced,
            df=self.full_model.design.shape[-1] - self.reduced_model.design.shape[-1]
        )

        return pval


def test(data, full_formula, reduced_formula, sample_description=None, model="nb_glm", close_sessions=True):
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

        X = data.X
    elif anndata is not None and isinstance(data, xr.Dataset):
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

        X = data["X"]
    else:
        if sample_description is None:
            raise ValueError(
                "Please specify `sample_description` or provide `data` as xarray.Dataset or anndata.AnnData " +
                "with corresponding sample annotations"
            )

        X = data

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

    if model == "nb_glm":
        import api.models.nb_glm as test_model

        logger.info("Estimating reduced model...")
        estim = test_model.Estimator(X, design_matrix=reduced_design)
        estim.initialize()
        estim.train(learning_rate=0.5, loss_history_size=200, stop_at_loss_change=0.05)
        estim.train(learning_rate=0.05, loss_history_size=200, stop_at_loss_change=0.05)
        if close_sessions:
            reduced_model = test_model.model_from_params(estim.export_params())
            estim.close_session()
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
        estim = test_model.Estimator(
            X,
            design_matrix=full_design,
            init_a_intercept=a_intercept,
            init_a_slopes=a_slopes,
            init_b_intercept=b_intercept,
            init_b_slopes=b_slopes,
        )
        estim.initialize()
        estim.train(learning_rate=0.5, loss_history_size=200, stop_at_loss_change=0.05)
        estim.train(learning_rate=0.05, loss_history_size=200, stop_at_loss_change=0.05)
        if close_sessions:
            full_model = test_model.model_from_params(estim.export_params())
            estim.close_session()
        else:
            full_model = estim
        logger.info("Estimation of full model ready")

        de_test = DifferentialExpressionTest(X, full_model, reduced_model)
        return de_test
