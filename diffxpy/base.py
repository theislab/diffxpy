import abc
import logging
from typing import Union, Dict, Tuple, List, Set

import pandas as pd

import numpy as np

# try:
import xarray as xr

# except ImportError:
#     xr = None

try:
    import anndata
except ImportError:
    anndata = None

import patsy
import sgdglm.data as data_utils
from sgdglm.api.models.glm import Model as GeneralizedLinearModel

from . import stats


class _Estimation(GeneralizedLinearModel, metaclass=abc.ABCMeta):
    """
    Dummy class specifying all needed methods / parameters necessary for a model
    fitted for DifferentialExpressionTest.
    Useful for type hinting.
    """
    
    @property
    @abc.abstractmethod
    def num_observations(self) -> int:
        pass
    
    @property
    @abc.abstractmethod
    def num_features(self) -> int:
        pass
    
    @property
    @abc.abstractmethod
    def features(self) -> np.ndarray:
        pass
    
    @property
    @abc.abstractmethod
    def observations(self) -> np.ndarray:
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


class _DifferentialExpressionTest(metaclass=abc.ABCMeta):
    """
    Dummy class specifying all needed methods / parameters necessary for DifferentialExpressionTest.
    Useful for type hinting. Structure:
    Methods which are called by constructor and which compute (corrected) p-values:
        __test()
        __correction()
    Accessor methods for important metrics which have to be extracted from estimated models:
        log_fold_change()
        reduced_model_gradient()
        full_model_gradient()
    Interface method which provides summary of results:
        results()
        plot()
    """
    
    def __init__(self):
        self.__pval = None
        self.__qval = None
    
    @property
    @abc.abstractmethod
    def gene_ids(self) -> np.ndarray:
        pass
    
    # @property
    # @abc.abstractmethod
    # def log_fold_change(self) -> np.ndarray:
    #     pass
    
    @property
    @abc.abstractmethod
    def reduced_model_gradient(self) -> np.ndarray:
        pass
    
    @property
    @abc.abstractmethod
    def full_model_gradient(self) -> np.ndarray:
        pass
    
    @abc.abstractmethod
    def _test(self, **kwargs) -> np.ndarray:
        pass
    
    @abc.abstractmethod
    def _correction(self, **kwargs) -> np.ndarray:
        pass
    
    @property
    def pval(self):
        if self.pval is None:
            self.__pval = self._test()
        return self.__pval
    
    @property
    def qval(self):
        if self.qval is None:
            self.__qval = self._correction()
        return self.__qval
    
    @property
    @abc.abstractmethod
    def summary(self, **kwargs) -> pd.DataFrame:
        pass
    
    @property
    @abc.abstractmethod
    def plot(self, **kwargs):
        pass


class _DiscreteDifferentialExpressionTest(_DifferentialExpressionTest, metaclass=abc.ABCMeta):
    """
    DifferentialExpressionTest for discrete design.
    """
    
    sample_description: pd.DataFrame
    full_design_info: patsy.design_info
    full_estim: _Estimation
    reduced_design_info: patsy.design_info
    reduced_estim: _Estimation
    
    def __init__(
            self,
            sample_description: pd.DataFrame,
            full_design_info: patsy.design_info,
            full_estim,
            reduced_design_info: patsy.design_info,
            reduced_estim
    ):
        super().__init__()
        self.sample_description = sample_description
        self.full_design_info = full_design_info
        self.full_estim = full_estim
        self.reduced_design_info = reduced_design_info
        self.reduced_estim = reduced_estim
    
    @property
    def gene_ids(self) -> np.ndarray:
        return self.full_estim.features
    
    @property
    def reduced_model_gradient(self):
        return self.reduced_estim.gradient
    
    @property
    def full_model_gradient(self):
        return self.full_estim.gradient
    
    def _log_fold_change(self, factors: Union[Dict, Tuple, Set, List], base=np.e):
        """
        Returns a xr.DataArray containing the locations for the different categories of the factors

        :param factors: the factors to select.
            E.g. `condition` or `batch` if formula would be `~ 1 + batch + condition`
        :param base: the log base to use; default is the natural logarithm
        :return: xr.DataArray
        """
        
        if not (isinstance(factors, list) or isinstance(factors, tuple) or isinstance(factors, set)):
            factors = {factors}
        if not isinstance(factors, set):
            factors = set(factors)
        
        di = self.full_design_info
        sample_description = self.sample_description[[f.name() for f in di.subset(factors).factor_infos]]
        dmat = self.full_estim.design_loc
        
        dmat, idx = np.unique(dmat, axis=0, return_index=True)
        sample_description = sample_description.iloc[idx].reset_index(drop=True)
        
        # factors = factors.intersection(di.term_names)
        
        cols = np.arange(len(di.column_names))
        sel = np.concatenate([cols[di.slice(f)] for f in factors], axis=0)
        neg_sel = np.zeros_like(cols).astype(bool)
        neg_sel[sel] = True
        
        dmat[:, neg_sel] = 0
        
        locations = self.full_estim.inverse_link_loc(dmat @ self.full_estim.par_link_loc)
        locations = np.log(locations, base=base)
        
        dist = np.expand_dims(locations, axis=0)
        dist = np.transpose(dist, [1, 0, 2]) - dist
        dist = xr.DataArray(dist, dims=("minuend", "subtrahend", "gene"))
        # retval = xr.Dataset({"logFC": retval})
        
        dist.coords["gene"] = self.gene_ids
        
        for col in sample_description:
            dist.coords["minuend_" + col] = (("minuend",), sample_description[col])
            dist.coords["subtrahend_" + col] = (("subtrahend",), sample_description[col])
        
        return dist
    
    def log_fold_change(self, base=np.e, return_type="dataframe"):
        """
        Calculates the pairwise log fold change(s) for this DifferentialExpressionTest.
        Returns some distance matrix representation of size (groups x groups x genes) where groups corresponds
        to the unique groups compared in this differential expression test.

        :param base: the log base to use; default is the natural logarithm
        :param return_type: Choose the return type.

            Possible values are:
                - "dataframe":
                    return a pandas.DataFrame with columns `gene`, `minuend_<group>`, `subtrahend_<group>` and `logFC`.
                - "xarray":
                    return a xarray.DataArray with dimensions `(minuend, subtrahend, gene)`

        :return: either pandas.DataFrame or xarray.DataArray
        """
        factors = set(self.full_design_info.term_names).difference(self.reduced_design_info.term_names)
        dists = self._log_fold_change(factors=factors, base=base)
        
        if return_type == "dataframe":
            df = dists.to_dataframe("logFC")
            df = df.reset_index().drop(["minuend", "subtrahend"], axis=1)
        else:
            return dists
    
    def log2_fold_change(self, return_type="dataframe"):
        """
        Calculates the pairwise log_2 fold change(s) for this DifferentialExpressionTest.

        See <self>.`log_fold_change` for futher details.

        :param return_type: Choose the return type.

            Possible values are "dataframe" and "xarray".
            See <self>.`log_fold_change` for futher details.

        :return: either pandas.DataFrame or xarray.DataArray
        """
        return self.log_fold_change(base=2, return_type=return_type)
    
    def log10_fold_change(self, return_type="dataframe"):
        """
        Calculates the log_10 fold change(s) for this DifferentialExpressionTest.

        See <self>.`log_fold_change` for futher details.

        :param return_type: Choose the return type.

            Possible values are "dataframe" and "xarray".
            See <self>.`log_fold_change` for futher details.

        :return: either pandas.DataFrame or xarray.DataArray
        """
        return self.log_fold_change(base=10, return_type=return_type)
    
    def locations(self):
        """
        Returns a pandas.DataFrame containing the locations for the different categories of the factors

        :return: pd.DataFrame
        """
        
        di = self.full_design_info
        sample_description = self.sample_description[[f.name() for f in di.factor_infos]]
        dmat = self.full_estim.design_loc
        
        dmat, idx = np.unique(dmat, axis=0, return_index=True)
        sample_description = sample_description.iloc[idx].reset_index(drop=True)
        
        retval = self.full_estim.inverse_link_loc(dmat @ self.full_estim.par_link_loc)
        retval = pd.DataFrame(retval, columns=self.full_estim.features)
        for col in sample_description:
            retval[col] = sample_description[col]
        
        retval = retval.set_index(list(sample_description.columns))
        
        return retval
    
    def scales(self):
        """
        Returns a pandas.DataFrame containing the scales for the different categories of the factors

        :return: pd.DataFrame
        """
        
        di = self.full_design_info
        sample_description = self.sample_description[[f.name() for f in di.factor_infos]]
        dmat = self.full_estim.design_scale
        
        dmat, idx = np.unique(dmat, axis=0, return_index=True)
        sample_description = sample_description.iloc[idx].reset_index(drop=True)
        
        retval = self.full_estim.inverse_link_scale(dmat @ self.full_estim.par_link_scale)
        retval = pd.DataFrame(retval, columns=self.full_estim.features)
        for col in sample_description:
            retval[col] = sample_description[col]
        
        retval = retval.set_index(list(sample_description.columns))
        
        return retval


class DifferentialExpressionTestSingle(_DiscreteDifferentialExpressionTest):
    """
    _DifferentialExpressionTest for unit_test with a single test per gene.
    The individual test object inherit directly from this class.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _test(self):
        pass
    
    def _correction(self):
        # code BH FDR correction here
        # self.qval = BH corrected PVAL
        pass
    
    def log_fold_change(self, base=np.e, return_type="dataframe"):
        """
        Calculates the log fold change between the two sample groups in this test.
        Returns some distance matrix representation of size `num_genes`.

        :param base: the log base to use; default is the natural logarithm
        :param return_type: Choose the return type.

            Possible values are:
                - "dataframe":
                    return a pandas.DataFrame with columns `gene`, `minuend_<group>`, `subtrahend_<group>` and `logFC`.
                - "xarray":
                    return a xarray.DataArray with dimensions `(gene, )`

        :return: either pandas.DataFrame or xarray.DataArray
        """
    
    def summary(self, **kwargs) -> pd.DataFrame:
        """
        Summarize differential expression results into an output table.
        """
        assert self.gene_ids is not None
        assert self.pval is not None
        assert self.qval is not None
        
        res = pd.DataFrame({
            "gene": self.gene_ids,
            "pval": self.pval,
            "qval": self.qval,
            "log2fc": self.log2_fold_change(),
            "grad": self.full_model_gradient,
            "grad_red": self.reduced_model_gradient,
        })
        
        return res
    
    def plot(self, **kwargs):
        """
        Create visual summary of top differentially expressed genes.
        """
        pass


class DifferentialExpressionTestMulti(_DiscreteDifferentialExpressionTest):
    """
    _DifferentialExpressionTest for unit_test with a multiple unit_test per gene.
    The individual test object inherit directly from this class.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def log_fold_change(self):
        pass
    
    def _test(self):
        pass
    
    def _correction(self):
        # code BH FDR correction here
        # self.qval = BH corrected PVAL
        pass
    
    def summary(self, **kwargs) -> pd.DataFrame:
        """
        Summarize differential expression results into an output table.
        """
        assert self.gene_ids is not None
        assert self.pval is not None
        assert self.qval is not None
        
        # TODO have to modify so that min max works on 3D arrays for pairwise
        res = pd.DataFrame({
            "gene": self.gene_ids,
            "pval": np.min(self.pval, axis=1),  # return minimal pval by gene
            "qval": np.min(self.qval, axis=1),  # return minimal qval by gene
            "lfc": np.max(self.log2FC(), axis=1),  # return maximal lfc by gene
            "grad": np.max(self.full_model_gradient(),
                           axis=1),  # return maximal gradient by gene
            "grad_red": np.max(self.reduced_model_gradient(),
                               axis=1),  # return maximal gradient by gene
        })
        
        return res
    
    def plot(self, **kwargs):
        """
        Create visual summary of top differentially expressed genes.
        """
        pass


class DifferentialExpressionTestWald(DifferentialExpressionTestSingle):
    """
    Single wald test per gene.
    """
    model_estim: _Estimation
    
    def __init__(self, model_estim: _Estimation):
        super().__init__()
        self.model_estim = model_estim
        self.test()
        self.bh_correction()
    
    def reduced_model_gradient(self):
        pass
    
    def full_model_gradient(self):
        return self.model_estim.gradient
    
    def log_fold_change(self):
        pass
    
    def _test(self, coef_loc_totest):
        full = np.sum(self.full_estim.log_probs(), axis=0)
        reduced = np.sum(self.reduced_estim.log_probs(), axis=0)
        
        # TODO extract MLE and std of model here.
        self.pval = stats.wald_test(theta_mle=self.model_estim., theta_sd=self.model_estim., theta0=0)


class DifferentialExpressionTestLRT(DifferentialExpressionTestSingle):
    """
    Single log-likelihood ratio test per gene.
    """
    full_estim: _Estimation
    reduced_estim: _Estimation
    
    def __init__(self, full_estim: _Estimation, reduced_estim: _Estimation):
        super().__init__()
        self.full_estim = full_estim
        self.reduced_estim = reduced_estim
        self._test()
        self._correction()
    
    def reduced_model_gradient(self):
        return self.reduced_estim.gradient
    
    def full_model_gradient(self):
        return self.full_estim.gradient
    
    def log_fold_change(self):
        pass
    
    def _test(self):
        full = np.sum(self.full_estim.log_probs(), axis=0)
        reduced = np.sum(self.reduced_estim.log_probs(), axis=0)
        
        self.pval = stats.likelihood_ratio_test(
            ll_full=full,
            ll_reduced=reduced,
            df=self.full_estim.design.shape[-1] - self.reduced_estim.design.shape[-1]
        )


class DifferentialExpressionTestTT(DifferentialExpressionTestSingle):
    """
    Single t-test test per gene.
    """
    
    def __init__(self, pval):
        super().__init__()
        self.pval = pval
        self._correction()


class DifferentialExpressionTestWilcoxon(DifferentialExpressionTestSingle):
    """
    Single wilcoxon rank sum test per gene.
    """
    
    def __init__(self, pval):
        super().__init__()
        self.pval = pval
        self._correction()


class DifferentialExpressionTestPairwise(DifferentialExpressionTestMulti):
    """
    Pairwise unit_test between more than 2 groups per gene.
    """
    
    def __init__(self, pval):
        super().__init__()
        self.pval = pval
        self._correction()
    
    def reduced_model_gradient(self):
        return self.grad_red
    
    def full_model_gradient(self):
        return self.grad_full
    
    def log_fold_change(self):
        return self.lfc


class DifferentialExpressionTestVsRest(DifferentialExpressionTestMulti):
    """
    Tests between between each group and the rest for more than 2 groups per gene.
    """
    
    def __init__(self, pval, grad_full, grad_red, lfc):
        super().__init__()
        self.pval = pval
        self.grad_full = grad_full
        self.grad_red = grad_red
        self._correction()
    
    def reduced_model_gradient(self):
        return self.grad_red
    
    def full_model_gradient(self):
        return self.grad_full
    
    def log_fold_change(self):
        return self.lfc


def test_lrt(
        data,
        full_formula_loc,
        reduced_formula_loc,
        full_formula_scale=None,
        reduced_formula_scale=None,
        sample_description=None,
        noise_model="nb",
        close_sessions=True
):
    """
    Perform log-likelihood ratio test for differential expression 
    between two groups on adata object for each gene.

    This function wraps the selected statistical test for the scenario of
    a two sample comparison. All unit_test offered in this wrapper
    test for the difference of the mean parameter of both samples.
        
    :param data
    :param full_formula_loc: formula
        Full model formula for location paramter model.
    :param reduced_formula_loc: formula
        Reduced model formula for location paramter model.
    :param full_formula_scale: formula
        Full model formula for scale paramter model.
        Will use same formula as for location parameter if not specified.
    :param reduced_formula_scale: formula
        Reduced model formula for scale paramter model.
        Will use same formula as for location parameter if not specified.
    :param method: str
        {'wald':default, 'lrt' 't-test', 'wilcoxon'} Statistical test to use.
    :param sample_description: 
    :param noise_model: str
        {'nb':default} Noise model to use in model-based unit_test.
    :param close_sessions: 
    """
    if full_formula_scale is None:
        full_formula_scale = full_formula_loc
    if reduced_formula_scale is None:
        reduced_formula_scale = reduced_formula_loc
    
    full_design_loc = None
    reduced_design_loc = None
    full_design_scale = None
    reduced_design_scale = None
    if anndata is not None and isinstance(data, anndata.AnnData):
        if sample_description is None:
            # Define location parameter models.
            full_design_loc = data_utils.design_matrix_from_anndata(
                dataset=data,
                formula=full_formula_loc
            )
            reduced_design_loc = data_utils.design_matrix_from_anndata(
                dataset=data,
                formula=full_design_loc.design_info.subset(reduced_formula_loc)
            )
            # Define scale parameter models.
            if full_formula_scale is None:
                full_design_scale = full_design_loc
            else:
                full_design_scale = data_utils.design_matrix_from_anndata(
                    dataset=data,
                    formula=full_formula_scale
                )
            if reduced_formula_scale is None:
                # Default to full model so that only the location parameters are tested.
                reduced_design_scale = full_design_loc
            else:
                reduced_design_scale = data_utils.design_matrix_from_anndata(
                    dataset=data,
                    formula=full_design_scale.design_info.subset(reduced_formula_scale)
                )
        
        # X = data.X
    elif xr is not None and isinstance(data, xr.Dataset):
        if sample_description is None:
            # Define location parameter models.
            full_design_loc = data_utils.design_matrix_from_xarray(
                dataset=data,
                dim="observations",
                formula=full_formula_loc
            )
            reduced_design_loc = data_utils.design_matrix_from_xarray(
                dataset=data,
                dim="observations",
                formula=full_design_loc.design_info.subset(reduced_formula_loc)
            )
            # Define scale parameter models.
            if full_formula_scale is None:
                full_design_scale = full_design_loc
            else:
                full_design_scale = data_utils.design_matrix_from_xarray(
                    dataset=data,
                    dim="observations",
                    formula=full_formula_scale
                )
            if reduced_formula_scale is None:
                # Default to full model so that only the location parameters are tested.
                reduced_design_scale = full_design_loc
            else:
                reduced_design_scale = data_utils.design_matrix_from_xarray(
                    dataset=data,
                    dim="observations",
                    formula=full_design_scale.design_info.subset(reduced_formula_scale)
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
        full_design_loc = data_utils.design_matrix(
            sample_description=sample_description,
            formula=full_formula_loc
        )
        reduced_design_loc = data_utils.design_matrix(
            sample_description=sample_description,
            formula=full_design_loc.design_info.subset(reduced_formula_loc)
        )
        if full_formula_scale is None:
            full_design_scale = full_design_loc
        else:
            full_design_scale = data_utils.design_matrix(
                sample_description=sample_description,
                formula=full_formula_scale
            )
        if reduced_formula_scale is None:
            # Default to full model so that only the location parameters are tested.
            reduced_design_scale = full_design_loc
        else:
            reduced_design_scale = data_utils.design_matrix(
                sample_description=sample_description,
                formula=full_design_scale.design_info.subset(reduced_formula_scale)
            )
    
    logger = logging.getLogger(__name__)
    
    if noise_model == "nb" or noise_model == "negative_binomial":
        import sgdglm.api.models.nb_glm as test_model
        
        logger.info("Estimating reduced model...")
        reduced_input_data = test_model.InputData.new(
            data=data, design_loc=reduced_design_loc, design_scale=reduced_design_scale
        )
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
        indices_loc = np.arange(len(full_design_loc.design_info.column_name_indexes))
        indices_loc = np.unique(np.concatenate([
            indices_loc[v] for i, v in full_design_loc.design_info.term_name_slices.items() if
            i in reduced_design_loc.design_info.term_names
        ]))
        
        # initialize `a` and `b` of the full model with estimated values from the reduced model
        shape_loc = list(reduced_model.a.shape)
        shape_loc[-2] = full_design_loc.shape[-1]
        
        location = np.tile(
            np.sqrt(np.nextafter(0, 1, dtype=reduced_model.a.dtype)),
            reps=shape_loc
        )
        scale = location.copy()
        
        location[indices_loc] = reduced_model.location
        location_intercept = location[[0]]
        location_slopes = location[1:]
        
        scale[indices_loc] = reduced_model.b
        scale_intercept = scale[[0]]
        scale_slopes = scale[1:]
        
        logger.info("Estimating full model...")
        full_input_data = test_model.InputData.new(
            data=data, design_loc=full_design_loc, design_scale=full_design_scale
        )
        estim = test_model.Estimator(
            input_data=full_input_data,
            init_model=reduced_model
        )
        estim.initialize()
        estim.train(learning_rate=0.5, loss_history_size=200, stop_at_loss_change=0.05)
        estim.train(learning_rate=0.05, loss_history_size=200, stop_at_loss_change=0.05)
        if close_sessions:
            full_model = estim.finalize()
        else:
            full_model = estim
        logger.info("Estimation of full model ready")
        
        de_test = DifferentialExpressionTestLRT(full_model, reduced_model)
    else:
        raise ValueError('base.test(): `noise_model` not recognized.')
    
    return de_test


def test_wald(data, formula_loc, coef_loc_totest, formula_scale=None,
              sample_description=None, noise_model="nb", close_sessions=True):
    """
    Perform log-likelihood ratio test for differential expression 
    between two groups on adata object for each gene.

    This function wraps the selected statistical test for the scenario of
    a two sample comparison. All unit_test offered in this wrapper
    test for the difference of the mean parameter of both samples.
        
    :param data
    :param formula_loc: formula
        Model formula for location paramter model.
    :param coef_loc_totest: str
        Model coefficient of location parameter model to test with Wald test.
    :param formula_scale: formula
        Model formula for scale paramter model.
    :param method: str
        {'wald':default, 'lrt' 't-test', 'wilcoxon'} Statistical test to use.
    :param sample_description: 
    :param noise_model: str
        {'nb':default} Noise model to use in model-based unit_test.
    :param close_sessions: 
    """
    design_loc = None
    design_scale = None
    if anndata is not None and isinstance(data, anndata.AnnData):
        if sample_description is None:
            # Define location parameter model.
            design_loc = data_utils.design_matrix_from_anndata(
                dataset=data,
                formula=formula_loc
            )
            # Define scale parameter model.
            if formula_scale is None:
                design_scale = design_loc
            else:
                design_scale = data_utils.design_matrix_from_anndata(
                    dataset=data,
                    formula=formula_scale
                )
        
        # X = data.X
    elif xr is not None and isinstance(data, xr.Dataset):
        if sample_description is None:
            # Define location parameter model.
            design_loc = data_utils.design_matrix_from_xarray(
                dataset=data,
                dim="observations",
                formula=full_formula_loc
            )
            # Define scale parameter model.
            if formula_scale is None:
                design_scale = design_loc
            else:
                design_scale = data_utils.design_matrix_from_xarray(
                    dataset=data,
                    dim="observations",
                    formula=formula_scale
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
        design_loc = data_utils.design_matrix(
            sample_description=sample_description,
            formula=formula_loc
        )
        if formula_scale is None:
            design_scale = design_loc
        else:
            design_scale = data_utils.design_matrix(
                sample_description=sample_description,
                formula=formula_scale
            )
    
    logger = logging.getLogger(__name__)
    
    if noise_model == "nb" or noise_model == "negative_binomial":
        import api.models.nb_glm as test_model
        
        logger.info("Estimating model...")
        input_data = test_model.InputData(data=data, design_loc=design_loc, design_scale=design_scale)
        estim = test_model.Estimator(input_data=input_data)
        estim.initialize()
        estim.train(learning_rate=0.5, loss_history_size=200, stop_at_loss_change=0.05)
        estim.train(learning_rate=0.05, loss_history_size=200, stop_at_loss_change=0.05)
        if close_sessions:
            model = estim.finalize()
        else:
            model = estim
        logger.info("Estimation model ready")
        
        de_test = DifferentialExpressionTestWald(model)
    else:
        raise ValueError('base.test(): `noise_model` not recognized.')
    
    return de_test


def test_t_test(data, grouping):
    """
    Perform Welch's t-test for differential expression 
    between two groups on adata object for each gene.

    :param data
    :param grouping: str
        Column in adata.obs which contains the split
        of observations into the two groups.
    """
    groups = np.unique(data.obs[grouping])  # TODO: make sure this is anndata
    pval = t_test_raw(
        x0=data[data.obs[grouping] == groups[0], :],
        x1=data[data.obs[grouping] == groups[1], :])
    de_test = DifferentialExpressionTestTT(model, pval)
    return de_test


def test_wilcoxon(data, grouping):
    """
    Perform Wilcoxon rank sum test for differential expression 
    between two groups on adata object for each gene.

    :param data
    :param grouping: str
        Column in adata.obs which contains the split
        of observations into the two groups.
    """
    groups = np.unique(data.obs[grouping])  # TODO: make sure this is anndata
    pval = wilcoxon(
        x0=data[data.obs[grouping] == groups[0], :],
        x1=data[data.obs[grouping] == groups[1], :])
    de_test = DifferentialExpressionTestTT(model, pval)
    return de_test


def two_sample(data, grouping: str, test=None, sample_description=None, noise_model=None, close_sessions=True):
    """
    Perform differential expression test between two groups on adata object
    for each gene.

    This function wraps the selected statistical test for the scenario of
    a two sample comparison. All unit_test offered in this wrapper
    test for the difference of the mean parameter of both samples.

    The exact unit_test are as follows (assuming the group labels
    are saved in a column named "group"):
    lrt(log-likelihood ratio test):
        Requires the fitting of 2 generalized linear models (full and reduced).
        full model location parameter: ~ 1+group
        full model scale parameter: ~ 1+group
        reduced model location parameter: ~ 1
        reduced model scale parameter: ~ 1+group
    Wald test:
        Requires the fitting of 1 generalized linear models.
        model location parameter: ~ 1+group
        model scale parameter: ~ 1+group
        Test the group coefficient of the location parameter model against 0.
    t-test:
        Doesn't require fitting of generalized linear models. 
        Welch's t-test between both observation groups.
    wilcoxon:
        Doesn't require fitting of generalized linear models.
        Wilcoxon rank sum (Mann-Whitney U) test between both observation groups.
        
    :param data
    :param grouping: str
        Column in adata.obs which contains the split
        of observations into the two groups.
    :param method: str
        {'wald':default, 'lrt' 't-test', 'wilcoxon'} Statistical test to use.
    :param sample_description: 
    :param noise_model: str
        {'nb':default} Noise model to use in model-based unit_test.
    :param close_sessions: 
    """
    if test in ['t-test', 'wilcoxon'] and noise_model is not None:
        raise ValueError('base.two_sample(): Do not specify `noise_model` if using test t-test or wilcoxon: ' +
                         'The t-test is based on a gaussian noise model and wilcoxon is model free.')
    
    # Set default test:
    if test is None:
        test = 'wald'
    
    if test == 'wald':
        # TODO handle formula syntax
        formula_loc = '~1+' + grouping
        formula_scale = '~1+' + grouping
        de_test = test_wald(data=data, formula_loc=formula_loc, formula_scale=formula_scale,
                            sample_description=sample_description, noise_model=noise_model,
                            close_sessions=close_sessions)
    elif test == 'lrt':
        # TODO handle formula syntax
        full_formula_loc = '~1+' + grouping
        red_formula_loc = '~1'
        full_formula_scale = '~1+' + grouping
        red_formula_scale = '~1+' + grouping
        de_test = stats.test_lrt(data=data, full_formula_loc=full_formula_loc, red_formula_loc=red_formula_loc,
                                 full_formula_scale=full_formula_scale, red_formula_scale=red_formula_scale,
                                 sample_description=sample_description, noise_model=noise_model,
                                 close_sessions=close_sessions)
    elif test == 't-test':
        de_test = stats.test_t_test(data=data)
    elif test == 'wilcoxon':
        de_test = stats.test_wilcoxon(data=data)
    else:
        raise ValueError('base.two_sample(): Parameter `test` not recognized.')
    
    return de_test


def test_pairwise(data, grouping: str, test='z-test', sample_description=None, noise_model=None, close_sessions=True):
    """
    Perform pairwise differential expression test between two groups on adata object
    for each gene for all combinations of pairs of groups.

    This function wraps the selected statistical test for the scenario of
    a two sample comparison. All unit_test offered in this wrapper
    test for the difference of the mean parameter of both samples. We note
    that the much more efficient default method is coefficient based
    and only requires one model fit.

    The exact unit_test are as follows (assuming the group labels
    are saved in a column named "group"), each test is executed
    on the subset of the data that only contains observations of a given
    pair of groups:
    lrt(log-likelihood ratio test):
        Requires the fitting of 2 generalized linear models (full and reduced).
        full model location parameter: ~ 1+group
        full model scale parameter: ~ 1+group
        reduced model location parameter: ~ 1
        reduced model scale parameter: ~ 1+group
    Wald test:
        Requires the fitting of 1 generalized linear models.
        model location parameter: ~ 1+group
        model scale parameter: ~ 1+group
        Test the group coefficient of the location parameter model against 0.
    t-test:
        Doesn't require fitting of generalized linear models. 
        Welch's t-test between both observation groups.
    wilcoxon:
        Doesn't require fitting of generalized linear models.
        Wilcoxon rank sum (Mann-Whitney U) test between both observation groups.
        
    :param data
    :param grouping: str
        Column in adata.obs which contains the split
        of observations into the two groups.
    :param method: str
        {'z-test':default,'wald', 'lrt' 't-test', 'wilcoxon'} Statistical test to use.
    :param sample_description: 
    :param noise_model: str
        {'nb':default} Noise model to use in model-based unit_test.
    :param close_sessions: 
    """
    # Do not store all models but only p-value and q-value matrix:
    # genes x groups x groups
    groups = np.unique(data.obs[grouping])
    pvals = np.zeros([data.X.shape[1], len(groups), len(groups)])
    
    if test == 'z-test':
        # TODO handle formula syntax
        formula_loc = '~1+' + grouping
        formula_scale = '~1+' + grouping
        design_loc = None
        design_scale = None
        if anndata is not None and isinstance(data, anndata.AnnData):
            if sample_description is None:
                # Define location parameter model.
                design_loc = data_utils.design_matrix_from_anndata(
                    dataset=data,
                    formula=formula_loc
                )
                # Define scale parameter model.
                if formula_scale is None:
                    design_scale = design_loc
                else:
                    design_scale = data_utils.design_matrix_from_anndata(
                        dataset=data,
                        formula=formula_scale
                    )
            
            # X = data.X
        elif xr is not None and isinstance(data, xr.Dataset):
            if sample_description is None:
                # Define location parameter model.
                design_loc = data_utils.design_matrix_from_xarray(
                    dataset=data,
                    dim="observations",
                    formula=full_formula_loc
                )
                # Define scale parameter model.
                if formula_scale is None:
                    design_scale = design_loc
                else:
                    design_scale = data_utils.design_matrix_from_xarray(
                        dataset=data,
                        dim="observations",
                        formula=formula_scale
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
            design_loc = data_utils.design_matrix(
                sample_description=sample_description,
                formula=formula_loc
            )
            if formula_scale is None:
                design_scale = design_loc
            else:
                design_scale = data_utils.design_matrix(
                    sample_description=sample_description,
                    formula=formula_scale
                )
        
        logger = logging.getLogger(__name__)
        
        if noise_model == "nb" or noise_model == "negative_binomial":
            import api.models.nb_glm as test_model
            
            logger.info("Estimating model...")
            input_data = test_model.InputData(data=data, design_loc=design_loc, design_scale=design_scale)
            estim = test_model.Estimator(input_data=input_data)
            estim.initialize()
            estim.train(learning_rate=0.5, loss_history_size=200, stop_at_loss_change=0.05)
            estim.train(learning_rate=0.05, loss_history_size=200, stop_at_loss_change=0.05)
            if close_sessions:
                model = estim.finalize()
            else:
                model = estim
            logger.info("Estimation model ready")
        else:
            raise ValueError('base.test(): `noise_model` not recognized.')
        
        ##TODO extract coefficients and standard deviation from model fit which are then used by ztest
        theta_mle = NA  # values of parameter estiamtes: genes x coefficient array with one coefficient per group
        theta_sd = NA  # standard deviation of estimates: genes x coefficient array with one coefficient per group
        for i, g1 in enumerate(groups):
            for j, g2 in enumerate(groups[(i + 1):]):
                pvals[:, i, j] = stats.two_coef_z_test(theta_mle0=theta_mle[:, i], theta_mle1=theta_mle[:, j],
                                                       theta_sd0=theta_sd[:, i], theta_sd1 == theta_sd[:, j])
    else:
        for i, g1 in enumerate(groups):
            for j, g2 in enumerate(groups[(i + 1):]):
                de_test_temp = stats.two_sample(data=data, grouping=grouping, test=test,
                                                sample_description=sample_description,
                                                noise_model=noise_model, close_sessions=close_sessions)
                pvals[:, i, j] = de_test_temp.pval
    
    # TODO extracrt lfc and gradients (if available)
    de_test = DifferentialExpressionTestPairwise(pval=pval, grad_full=None, grad_red=None, lfc=None)
    return de_test


def test_vsrest(data, grouping: str, test='fast-wald', sample_description=None, noise_model=None, close_sessions=True):
    """
    Perform pairwise differential expression test between two groups on adata object
    for each gene for each groups versus the rest of the data set.

    This function wraps the selected statistical test for the scenario of
    a two sample comparison. All unit_test offered in this wrapper
    test for the difference of the mean parameter of both samples. We note
    that the much more efficient default method is coefficient based
    and only requires one model fit.

    The exact unit_test are as follows (assuming the group labels
    are saved in a column named "group"), each test is executed
    on the entire data and the labels are modified so that the target group
    is one group and the remaining groups are allocated to the second reference
    group):
    lrt(log-likelihood ratio test):
        Requires the fitting of 2 generalized linear models (full and reduced).
        full model location parameter: ~ 1+group
        full model scale parameter: ~ 1+group
        reduced model location parameter: ~ 1
        reduced model scale parameter: ~ 1+group
    Wald test:
        Requires the fitting of 1 generalized linear models.
        model location parameter: ~ 1+group
        model scale parameter: ~ 1+group
        Test the group coefficient of the location parameter model against 0.
    t-test:
        Doesn't require fitting of generalized linear models. 
        Welch's t-test between both observation groups.
    wilcoxon:
        Doesn't require fitting of generalized linear models.
        Wilcoxon rank sum (Mann-Whitney U) test between both observation groups.
    fast-wald:
        Requires the fitting of 1 generalized linear models.
        model location parameter: ~ group
        model scale parameter: ~ group
        Test each groups location coefficient against the overall average expression.
        
    :param data
    :param grouping: str
        Column in adata.obs which contains the split
        of observations into the two groups.
    :param method: str
        {'fast-wald':default,'wald', 'lrt' 't-test', 'wilcoxon'} Statistical test to use.
    :param sample_description: 
    :param noise_model: str
        {'nb':default} Noise model to use in model-based unit_test.
    :param close_sessions: 
    """
    # Do not store all models but only p-value and q-value matrix:
    # genes x groups
    groups = np.unique(data.obs[grouping])
    pvals = np.zeros([data.X.shape[1], len(groups)])
    
    if test == 'fast-wald':
        # TODO handle formula syntax
        formula_loc = '~1+' + grouping
        formula_scale = '~1+' + grouping
        design_loc = None
        design_scale = None
        if anndata is not None and isinstance(data, anndata.AnnData):
            if sample_description is None:
                # Define location parameter model.
                design_loc = data_utils.design_matrix_from_anndata(
                    dataset=data,
                    formula=formula_loc
                )
                # Define scale parameter model.
                if formula_scale is None:
                    design_scale = design_loc
                else:
                    design_scale = data_utils.design_matrix_from_anndata(
                        dataset=data,
                        formula=formula_scale
                    )
            
            # X = data.X
        elif xr is not None and isinstance(data, xr.Dataset):
            if sample_description is None:
                # Define location parameter model.
                design_loc = data_utils.design_matrix_from_xarray(
                    dataset=data,
                    dim="observations",
                    formula=full_formula_loc
                )
                # Define scale parameter model.
                if formula_scale is None:
                    design_scale = design_loc
                else:
                    design_scale = data_utils.design_matrix_from_xarray(
                        dataset=data,
                        dim="observations",
                        formula=formula_scale
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
            design_loc = data_utils.design_matrix(
                sample_description=sample_description,
                formula=formula_loc
            )
            if formula_scale is None:
                design_scale = design_loc
            else:
                design_scale = data_utils.design_matrix(
                    sample_description=sample_description,
                    formula=formula_scale
                )
        
        logger = logging.getLogger(__name__)
        
        if noise_model == "nb" or noise_model == "negative_binomial":
            import api.models.nb_glm as test_model
            
            logger.info("Estimating model...")
            input_data = test_model.InputData(data=data, design_loc=design_loc, design_scale=design_scale)
            estim = test_model.Estimator(input_data=input_data)
            estim.initialize()
            estim.train(learning_rate=0.5, loss_history_size=200, stop_at_loss_change=0.05)
            estim.train(learning_rate=0.05, loss_history_size=200, stop_at_loss_change=0.05)
            if close_sessions:
                model = estim.finalize()
            else:
                model = estim
            logger.info("Estimation model ready")
        else:
            raise ValueError('base.test(): `noise_model` not recognized.')
        
        ##TODO extract coefficients and standard deviation from model fit which are then used by ztest
        theta_mle = NA  # values of parameter estiamtes: genes x coefficient array with one coefficient per group
        theta_sd = NA  # standard deviation of estimates: genes x coefficient array with one coefficient per group
        ave_expr = np.mean(data.X, axis=0).flatten()
        for i, g1 in enumerate(groups):
            pvals[:, i] = stats.wald_test(theta_mle=theta_mle[:, i], theta_sd[:, i], theta0=ave_expr)
    else:
        for i, g1 in enumerate(groups):
            # TODO adjust group allocation that group g1 is tested versus union of all other groups.
            de_test_temp = stats.two_sample(data=data, grouping=grouping, test=test,
                                            sample_description=sample_description,
                                            noise_model=noise_model, close_sessions=close_sessions)
            pvals[:, i] = de_test_temp.pval
    
    # TODO extracrt lfc and gradients (if available)
    de_test = DifferentialExpressionTestVsRest(pval=pval, grad_full=None, grad_red=None, lfc=None)
    return de_test
