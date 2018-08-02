import abc
import logging
from typing import Union, Dict, Tuple, List, Set, Callable

import pandas as pd

import numpy as np
import scipy.sparse

import dask
import xarray as xr

try:
    import anndata
except ImportError:
    anndata = None

import patsy
import batchglm.data as data_utils
from batchglm.api.models.glm import Model as GeneralizedLinearModel

from . import stats
from . import correction

logger = logging.getLogger(__name__)


def _dmat_unique(dmat, sample_description):
    dmat, idx = np.unique(dmat, axis=0, return_index=True)
    sample_description = sample_description.iloc[idx].reset_index(drop=True)
    
    return dmat, sample_description


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
    def fisher_loc(self, **kwargs) -> np.ndarray:
        pass
    
    @property
    @abc.abstractmethod
    def fisher_scale(self, **kwargs) -> np.ndarray:
        pass


class _DifferentialExpressionTest(metaclass=abc.ABCMeta):
    """
    Dummy class specifying all needed methods / parameters necessary for DifferentialExpressionTest.
    Useful for type hinting. Structure:
    Methods which are called by constructor and which compute (corrected) p-values:
        _test()
        _correction()
    Accessor methods for important metrics which have to be extracted from estimated models:
        log_fold_change()
        reduced_model_gradient()
        full_model_gradient()
    Interface method which provides summary of results:
        results()
        plot()
    """
    
    def __init__(self):
        self._pval = None
        self._qval = None
    
    @property
    @abc.abstractmethod
    def gene_ids(self) -> np.ndarray:
        pass
    
    @abc.abstractmethod
    def log_fold_change(self, base=np.e, **kwargs):
        pass
    
    def log2_fold_change(self, **kwargs):
        """
        Calculates the pairwise log_2 fold change(s) for this DifferentialExpressionTest.
        """
        return self.log_fold_change(base=2, **kwargs)
    
    def log10_fold_change(self, **kwargs):
        """
        Calculates the log_10 fold change(s) for this DifferentialExpressionTest.
        """
        return self.log_fold_change(base=10, **kwargs)
    
    def _test(self, **kwargs) -> np.ndarray:
        pass
    
    def _correction(self, method) -> np.ndarray:
        """
        Performs multiple testing corrections available in statsmodels.stats.multitest.multipletests()
        on self.pval.

        :param method: Multiple testing correction method.
            Browse available methods in the annotation of statsmodels.stats.multitest.multipletests().
        """
        return correction.correct(pvals=self.pval, method=method)
    
    @property
    def pval(self):
        if self._pval is None:
            self._pval = self._test()
        return self._pval
    
    @property
    def qval(self, method="fdr_bh"):
        if self._qval is None:
            self._qval = self._correction(method=method)
        return self._qval
    
    @property
    @abc.abstractmethod
    def summary(self, **kwargs) -> pd.DataFrame:
        pass
    
    # @abc.abstractmethod
    def plot(self, **kwargs):
        pass


class _DifferentialExpressionTestSingle(_DifferentialExpressionTest, metaclass=abc.ABCMeta):
    """
    _DifferentialExpressionTest for unit_test with a single test per gene.
    The individual test object inherit directly from this class.

    All implementations of this class should return one p-value and one fold change per gene.
    """
    
    def summary(self, **kwargs) -> pd.DataFrame:
        """
        Summarize differential expression results into an output table.
        """
        assert self.gene_ids is not None
        
        res = pd.DataFrame({
            "gene": self.gene_ids,
            "pval": self.pval,
            "qval": self.qval,
            "log2fc": self.log2_fold_change()
        })
        
        return res


class DifferentialExpressionTestLRT(_DifferentialExpressionTestSingle):
    """
    Single log-likelihood ratio test per gene.
    """
    
    sample_description: pd.DataFrame
    full_design_info: patsy.design_info
    full_estim: _Estimation
    reduced_design_info: patsy.design_info
    reduced_estim: _Estimation
    
    def __init__(
            self,
            sample_description: pd.DataFrame,
            full_design_loc_info: patsy.design_info,
            full_estim,
            reduced_design_loc_info: patsy.design_info,
            reduced_estim
    ):
        super().__init__()
        self.sample_description = sample_description
        self.full_design_info = full_design_loc_info
        self.full_estim = full_estim
        self.reduced_design_info = reduced_design_loc_info
        self.reduced_estim = reduced_estim
    
    @property
    def gene_ids(self) -> np.ndarray:
        return np.asarray(self.full_estim.features)
    
    @property
    def reduced_model_gradient(self):
        return self.reduced_estim.gradient
    
    @property
    def full_model_gradient(self):
        return self.full_estim.gradient
    
    def _test(self):
        full = np.sum(self.full_estim.log_probs(), axis=0)
        reduced = np.sum(self.reduced_estim.log_probs(), axis=0)
        
        return stats.likelihood_ratio_test(
            ll_full=full,
            ll_reduced=reduced,
            df_full=self.full_estim.design_loc.shape[-1],
            df_reduced=self.reduced_estim.design_loc.shape[-1]
        )
    
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
        
        # make rows unique
        dmat, sample_description = _dmat_unique(dmat, sample_description)
        
        # factors = factors.intersection(di.term_names)
        
        # select the columns of the factors
        cols = np.arange(len(di.column_names))
        sel = np.concatenate([cols[di.slice(f)] for f in factors], axis=0)
        neg_sel = np.ones_like(cols).astype(bool)
        neg_sel[sel] = False
        
        # overwrite all columns which are not specified by the factors with 0
        dmat[:, neg_sel] = 0
        
        # make the design matrix + sample description unique again
        dmat, sample_description = _dmat_unique(dmat, sample_description)
        
        locations = self.full_estim.inverse_link_loc(dmat @ self.full_estim.par_link_loc)
        locations = np.log(locations) / np.log(base)
        
        dist = np.expand_dims(locations, axis=0)
        dist = np.transpose(dist, [1, 0, 2]) - dist
        dist = xr.DataArray(dist, dims=("minuend", "subtrahend", "gene"))
        # retval = xr.Dataset({"logFC": retval})
        
        dist.coords["gene"] = self.gene_ids
        
        for col in sample_description:
            dist.coords["minuend_" + col] = (("minuend",), sample_description[col])
            dist.coords["subtrahend_" + col] = (("subtrahend",), sample_description[col])
        
        # # If this is a pairwise comparison, return only one fold change per gene
        # if dist.shape[:2] == (2, 2):
        #     dist = dist[1, 0]
        
        return dist
    
    def log_fold_change(self, base=np.e, return_type="vector"):
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
        
        if return_type == "dataframe":
            dists = self._log_fold_change(factors=factors, base=base)
            
            df = dists.to_dataframe("logFC")
            df = df.reset_index().drop(["minuend", "subtrahend"], axis=1, errors="ignore")
            return df
        elif return_type == "vector":
            if len(factors) > 1 or self.sample_description[list(factors)].drop_duplicates().shape[0] != 2:
                return None
            else:
                dists = self._log_fold_change(factors=factors, base=base)
                return dists[1, 0].values
        else:
            dists = self._log_fold_change(factors=factors, base=base)
            return dists
    
    def locations(self):
        """
        Returns a pandas.DataFrame containing the locations for the different categories of the factors

        :return: pd.DataFrame
        """
        
        di = self.full_design_info
        sample_description = self.sample_description[[f.name() for f in di.factor_infos]]
        dmat = self.full_estim.design_loc
        
        dmat, sample_description = _dmat_unique(dmat, sample_description)
        
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
        
        dmat, sample_description = _dmat_unique(dmat, sample_description)
        
        retval = self.full_estim.inverse_link_scale(dmat @ self.full_estim.par_link_scale)
        retval = pd.DataFrame(retval, columns=self.full_estim.features)
        for col in sample_description:
            retval[col] = sample_description[col]
        
        retval = retval.set_index(list(sample_description.columns))
        
        return retval
    
    def summary(self, **kwargs) -> pd.DataFrame:
        """
        Summarize differential expression results into an output table.
        """
        res = super().summary(**kwargs)
        res["grad"] = self.full_model_gradient.data
        res["grad_red"] = self.reduced_model_gradient.data
        
        return res


class DifferentialExpressionTestWald(_DifferentialExpressionTestSingle):
    """
    Single wald test per gene.
    """
    
    model_estim: _Estimation
    coef_loc_totest: int
    
    def __init__(self, model_estim: _Estimation, col_index):
        super().__init__()
        self.model_estim = model_estim
        self.coef_loc_totest = col_index
        p = self.pval
        q = self.qval
    
    @property
    def gene_ids(self) -> np.ndarray:
        return np.asarray(self.model_estim.features)
    
    @property
    def model_gradient(self):
        return self.model_estim.gradient
    
    def log_fold_change(self, base=np.e, **kwargs):
        """
        Returns one fold change per gene
        """
        design = np.unique(self.model_estim.design_loc, axis=0)
        dmat = np.zeros_like(design)
        dmat[:, self.coef_loc_totest] = design[:, self.coef_loc_totest]
        
        loc = dmat @ self.model_estim.par_link_loc[self.coef_loc_totest]
        return loc[1] - loc[0]
    
    def _test(self):
        theta_mle = self.model_estim.par_link_loc[self.coef_loc_totest]
        # standard deviation of estimates: genes x coefficient array with one coefficient per group
        # $\text{SE}(\hat{\theta}_{ML}) = \frac{1}{Fisher(\hat{\theta}_{ML})}$
        theta_sd = 1 / np.sqrt(
            np.asarray(self.model_estim.fisher_loc[self.coef_loc_totest])
        )
        return stats.wald_test(theta_mle=theta_mle, theta_sd=theta_sd, theta0=0)
    
    def summary(self, **kwargs) -> pd.DataFrame:
        """
        Summarize differential expression results into an output table.
        """
        res = super().summary(**kwargs)
        res["grad"] = self.model_gradient.data
        
        return res


class DifferentialExpressionTestTT(_DifferentialExpressionTestSingle):
    """
    Single t-test test per gene.
    """
    
    def __init__(self, gene_ids, pval, logfc):
        super().__init__()
        self._gene_ids = np.asarray(gene_ids)
        self._logfc = logfc
        self._pval = pval
        
        q = self.qval
    
    @property
    def gene_ids(self) -> np.ndarray:
        return self._gene_ids
    
    def log_fold_change(self, base=np.e, **kwargs):
        """
        Returns one fold change per gene
        """
        if base == np.e:
            return self._logfc
        else:
            return self._logfc / np.log(base)


class DifferentialExpressionTestWilcoxon(_DifferentialExpressionTestSingle):
    """
    Single wilcoxon rank sum test per gene.
    """
    
    def __init__(self, gene_ids, pval, logfc):
        super().__init__()
        self._gene_ids = np.asarray(gene_ids)
        self._logfc = logfc
        self._pval = pval
        
        q = self.qval
    
    @property
    def gene_ids(self) -> np.ndarray:
        return self._gene_ids
    
    def log_fold_change(self, base=np.e, **kwargs):
        """
        Returns one fold change per gene
        """
        if base == np.e:
            return self._logfc
        else:
            return self._logfc / np.log(base)


class _DifferentialExpressionTestMulti(_DifferentialExpressionTest, metaclass=abc.ABCMeta):
    """
    _DifferentialExpressionTest for unit_test with a multiple unit_test per gene.
    The individual test object inherit directly from this class.
    """
    
    def summary(self, **kwargs) -> pd.DataFrame:
        """
        Summarize differential expression results into an output table.
        """
        assert self.gene_ids is not None
        
        # calculate maximum logFC of lower triangular fold change matrix
        raw_logfc = self.log2_fold_change(return_type="xarray")
        argm = np.argmax(raw_logfc, axis=0)
        args = np.argmax(raw_logfc[argm], axis=0)
        argm = argm[args]
        logfc = raw_logfc[argm, args] * np.where(argm > args, 1, -1)
        
        res = pd.DataFrame({
            "gene": self.gene_ids,
            # return minimal pval by gene:
            "pval": np.min(self.pval, axis=1),
            # return minimal qval by gene:
            "qval": np.min(self.qval, axis=1),
            # return maximal logFC by gene:
            "log2fc": logfc
        })
        
        return res


class DifferentialExpressionTestPairwise(_DifferentialExpressionTestMulti):
    """
    Pairwise unit_test between more than 2 groups per gene.
    """
    
    def __init__(self, gene_ids, pval, logfc):
        super().__init__()
        self._gene_ids = np.asarray(gene_ids)
        self._logfc = logfc
        self._pval = pval
        
        q = self.qval
    
    @property
    def gene_ids(self) -> np.ndarray:
        return self._gene_ids
    
    def log_fold_change(self, base=np.e, **kwargs):
        """
        Returns matrix of fold changes per gene
        """
        if base == np.e:
            return self._logfc
        else:
            return self._logfc / np.log(base)


class DifferentialExpressionTestVsRest(_DifferentialExpressionTestMulti):
    """
    Tests between between each group and the rest for more than 2 groups per gene.
    """
    
    def __init__(self, gene_ids, pval, logfc):
        super().__init__()
        self._gene_ids = np.asarray(gene_ids)
        self._pval = pval
        self._logfc = logfc
        
        q = self.qval
    
    @property
    def gene_ids(self) -> np.ndarray:
        return self._gene_ids
    
    def log_fold_change(self, base=np.e, **kwargs):
        if base == np.e:
            return self._logfc
        else:
            return self._logfc / np.log(base)


def _parse_gene_names(data, gene_names):
    if gene_names is None:
        if anndata is not None and isinstance(data, anndata.AnnData):
            gene_names = data.var_names
        elif isinstance(data, xr.DataArray):
            gene_names = data["features"]
        elif isinstance(data, xr.Dataset):
            gene_names = data["features"]
        else:
            raise ValueError("Missing gene names")
    
    return np.asarray(gene_names)


def _parse_data(data, gene_names):
    X = data_utils.xarray_from_data(data, dims=("observations", "features"))
    if gene_names is not None:
        X.coords["features"] = gene_names
    
    return X


def _parse_sample_description(data, sample_description=None) -> pd.DataFrame:
    if sample_description is None:
        if anndata is not None and isinstance(data, anndata.AnnData):
            sample_description = data_utils.sample_description_from_anndata(
                dataset=data,
            )
        elif isinstance(data, xr.Dataset):
            sample_description = data_utils.sample_description_from_xarray(
                dataset=data,
                dim="observations",
            )
        else:
            raise ValueError(
                "Please specify `sample_description` or provide `data` as xarray.Dataset or anndata.AnnData " +
                "with corresponding sample annotations"
            )
    return sample_description


def _fit(
        noise_model,
        data,
        design_loc,
        design_scale,
        init_model=None,
        gene_names=None,
        batch_size: int = None,
        training_strategy: Union[str, List[Dict[str, object]], Callable] = "AUTO",
        close_session=True
):
    """
    :param noise_model: str, noise model to use in model-based unit_test. Possible options:
        
        - 'nb': default
    :param batch_size: the batch size to use for the estimator
    :param training_strategy: {str, function, list} training strategy to use. Can be:

        - str: will use Estimator.TrainingStrategy[training_strategy] to train
        - function: Can be used to implement custom training function will be called as
          `training_strategy(estimator)`.
        - list of keyword dicts containing method arguments: Will call Estimator.train() once with each dict of
          method arguments.
          
          Example:
          
          .. code-block:: python
          
              [
                {"learning_rate": 0.5, },
                {"learning_rate": 0.05, },
              ]

          This will run training first with learning rate = 0.5 and then with learning rate = 0.05.
    :param close_session: If True, will finalize the estimator. Otherwise, return the estimator itself.
    """
    if noise_model == "nb" or noise_model == "negative_binomial":
        import batchglm.api.models.nb_glm as test_model
        
        logger.info("Estimating model...")
        input_data = test_model.InputData.new(
            data=data,
            design_loc=design_loc,
            design_scale=design_scale,
            feature_names=gene_names
        )
        
        constructor_args = {}
        if batch_size is not None:
            constructor_args["batch_size"] = batch_size
        estim = test_model.Estimator(input_data=input_data, init_model=init_model, **constructor_args)
        
        estim.initialize()
        
        # training:
        if callable(training_strategy):
            # call training_strategy if it is a function
            training_strategy(estim)
        else:
            estim.train_sequence(training_strategy)
        
        if close_session:
            model = estim.finalize()
        else:
            model = estim
        logger.info("Estimating model ready")
    
    else:
        raise ValueError('base.test(): `noise_model` not recognized.')
    
    return model


def test_lrt(
        data,
        reduced_formula: str = None,
        full_formula: str = None,
        reduced_formula_loc: str = None,
        full_formula_loc: str = None,
        reduced_formula_scale: str = None,
        full_formula_scale: str = None,
        gene_names=None,
        sample_description: pd.DataFrame = None,
        noise_model="nb",
        batch_size: int = None,
        training_strategy: Union[str, List[Dict[str, object]], Callable] = "AUTO",
):
    """
    Perform log-likelihood ratio test for differential expression 
    between two groups on adata object for each gene.

    This function wraps the selected statistical test for the scenario of
    a two sample comparison. All unit_test offered in this wrapper
    test for the difference of the mean parameter of both samples.
        
    :param data: input data
    :param reduced_formula: formula
        Reduced model formula for location and scale parameter models.
    :param full_formula: formula
        Full model formula for location and scale parameter models.
    :param reduced_formula_loc: formula
        Reduced model formula for location and scale parameter models.
        If not specified, `reduced_formula` will be used instead.
    :param full_formula_loc: formula
        Full model formula for location parameter model.
        If not specified, `full_formula` will be used instead.
    :param reduced_formula_scale: formula
        Reduced model formula for scale parameter model.
        If not specified, `reduced_formula` will be used instead.
    :param full_formula_scale: formula
        Full model formula for scale parameter model.
        If not specified, `reduced_formula_scale` will be used instead.
    :param gene_names: optional list/array of gene names which will be used if `data` does not implicitly store these
    :param sample_description: optional pandas.DataFrame containing sample annotations
    :param noise_model: str, noise model to use in model-based unit_test. Possible options:
        
        - 'nb': default
    :param batch_size: the batch size to use for the estimator
    :param training_strategy: {str, function, list} training strategy to use. Can be:

        - str: will use Estimator.TrainingStrategy[training_strategy] to train
        - function: Can be used to implement custom training function will be called as
          `training_strategy(estimator)`.
        - list of keyword dicts containing method arguments: Will call Estimator.train() once with each dict of
          method arguments.
          
          Example:
          
          .. code-block:: python
          
              [
                {"learning_rate": 0.5, },
                {"learning_rate": 0.05, },
              ]

          This will run training first with learning rate = 0.5 and then with learning rate = 0.05.
    """
    
    if full_formula_loc is None:
        full_formula_loc = full_formula
    if reduced_formula_loc is None:
        reduced_formula_loc = reduced_formula
    if full_formula_scale is None:
        full_formula_scale = full_formula
    if reduced_formula_scale is None:
        reduced_formula_scale = reduced_formula
    
    X = _parse_data(data, gene_names)
    gene_names = _parse_gene_names(data, gene_names)
    sample_description = _parse_sample_description(data, sample_description)
    
    full_design_loc = data_utils.design_matrix(
        sample_description=sample_description, formula=full_formula_loc)
    reduced_design_loc = data_utils.design_matrix(
        sample_description=sample_description, formula=reduced_formula_loc)
    full_design_scale = data_utils.design_matrix(
        sample_description=sample_description, formula=full_formula_scale)
    reduced_design_scale = data_utils.design_matrix(
        sample_description=sample_description, formula=reduced_formula_scale)
    
    reduced_model = _fit(
        noise_model=noise_model,
        data=X,
        design_loc=reduced_design_loc,
        design_scale=reduced_design_scale,
        gene_names=gene_names,
        batch_size=batch_size,
        training_strategy=training_strategy,
    )
    full_model = _fit(
        noise_model=noise_model,
        data=X,
        design_loc=full_design_loc,
        design_scale=full_design_scale,
        gene_names=gene_names,
        init_model=reduced_model,
        batch_size=X.shape[0],  # workaround: batch_size=num_observations
        training_strategy=training_strategy,
    )
    
    de_test = DifferentialExpressionTestLRT(
        sample_description=sample_description,
        full_design_loc_info=full_design_loc.design_info,
        full_estim=full_model,
        reduced_design_loc_info=reduced_design_loc.design_info,
        reduced_estim=reduced_model,
    )
    
    return de_test


def test_wald_loc(
        data,
        factor_loc_totest: str,
        coef_to_test: object = None,  # e.g. coef_to_test="B"
        formula: str = None,
        formula_loc: str = None,
        formula_scale: str = None,
        gene_names: str = None,
        sample_description: pd.DataFrame = None,
        noise_model: str = "nb",
        batch_size: int = None,
        training_strategy: Union[str, List[Dict[str, object]], Callable] = "AUTO",
):
    """
    Perform log-likelihood ratio test for differential expression
    between two groups on adata object for each gene.

    This function wraps the selected statistical test for the scenario of
    a two sample comparison. All unit_test offered in this wrapper
    test for the difference of the mean parameter of both samples.
    
    :param data: input data
    :param formula: formula
        model formula for location and scale parameter models.
    :param formula_loc: formula
        model formula for location and scale parameter models.
        If not specified, `formula` will be used instead.
    :param formula_scale: formula
        model formula for scale parameter model.
        If not specified, `formula` will be used instead.
    :param factor_loc_totest: str
        Factor of formula to test with Wald test.
        E.g. "condition" if formula_loc would be "~ 1 + batch + condition"
    :param coef_to_test: If there are more than two groups specified by `factor_loc_totest`,
        this parameter allows to specify the group which should be tested
    :param gene_names: optional list/array of gene names which will be used if `data` does not implicitly store these
    :param sample_description: optional pandas.DataFrame containing sample annotations
    :param noise_model: str, noise model to use in model-based unit_test. Possible options:
        
        - 'nb': default
    :param batch_size: the batch size to use for the estimator
    :param training_strategy: {str, function, list} training strategy to use. Can be:

        - str: will use Estimator.TrainingStrategy[training_strategy] to train
        - function: Can be used to implement custom training function will be called as
          `training_strategy(estimator)`.
        - list of keyword dicts containing method arguments: Will call Estimator.train() once with each dict of
          method arguments.
          
          Example:
          
          .. code-block:: python
          
              [
                {"learning_rate": 0.5, },
                {"learning_rate": 0.05, },
              ]

          This will run training first with learning rate = 0.5 and then with learning rate = 0.05.
    """
    
    if formula_loc is None:
        formula_loc = formula
    if formula_scale is None:
        formula_scale = formula
    assert formula_scale is not None and formula_loc is not None, "Missing formula!"
    
    X = _parse_data(data, gene_names)
    gene_names = _parse_gene_names(data, gene_names)
    sample_description = _parse_sample_description(data, sample_description)
    
    design_loc = data_utils.design_matrix(
        sample_description=sample_description, formula=formula_loc)
    design_scale = data_utils.design_matrix(
        sample_description=sample_description, formula=formula_scale)
    
    col_slice = np.arange(design_loc.shape[-1])[design_loc.design_info.slice(factor_loc_totest)]
    assert col_slice.size > 0, "Could not find any matching columns!"
    
    if col_slice.size == 1:
        # only one column possible
        col_index = col_slice[0]
    else:
        samples = sample_description[factor_loc_totest].astype(type(coef_to_test)) == coef_to_test
        one_cols = np.where(design_loc[samples][:, col_slice][0] == 1)
        if one_cols.size == 0:
            # there is no such column; modify design matrix to create one
            col_index = col_slice[0]
            design_loc[:, col_index] = np.where(samples, 1, 0)
        else:
            # use the one_column as col_index
            col_index = one_cols[0]
    
    model = _fit(
        noise_model=noise_model,
        data=X,
        design_loc=design_loc,
        design_scale=design_scale,
        gene_names=gene_names,
        batch_size=batch_size,
        training_strategy=training_strategy,
    )
    
    de_test = DifferentialExpressionTestWald(model, col_index=col_index)
    
    return de_test


def _parse_grouping(data, sample_description, grouping):
    if isinstance(grouping, str):
        sample_description = _parse_sample_description(data, sample_description)
        grouping = sample_description[grouping]
    return np.squeeze(np.asarray(grouping))


def _split_X(data, grouping):
    groups = np.unique(grouping)
    x0 = data[grouping == groups[0], :]
    x1 = data[grouping == groups[1], :]
    return x0, x1


def test_t_test(
        data,
        grouping,
        gene_names=None,
        sample_description=None
):
    """
    Perform Welch's t-test for differential expression 
    between two groups on adata object for each gene.

    :param data: input data
    :param grouping: str, array
    
        - column in data.obs/sample_description which contains the split of observations into the two groups.
        - array of length `num_observations` containing group labels
    :param gene_names: optional list/array of gene names which will be used if `data` does not implicitly store these
    :param sample_description: optional pandas.DataFrame containing sample annotations
    """
    gene_names = _parse_gene_names(data, gene_names)
    grouping = _parse_grouping(data, grouping, sample_description)
    x0, x1 = _split_X(data, grouping)
    
    pval = stats.t_test_raw(x0=x0.data, x1=x1.data)
    logfc = np.log(np.mean(x1, axis=0)) - np.log(np.mean(x0, axis=0)).data
    
    de_test = DifferentialExpressionTestTT(
        gene_ids=gene_names,
        pval=pval,
        logfc=logfc,
    )
    
    return de_test


def test_wilcoxon(
        data,
        grouping,
        gene_names=None,
        sample_description=None
):
    """
    Perform Wilcoxon rank sum test for differential expression 
    between two groups on adata object for each gene.

    :param data: input data
    :param grouping: str, array
    
        - column in data.obs/sample_description which contains the split of observations into the two groups.
        - array of length `num_observations` containing group labels
    :param gene_names: optional list/array of gene names which will be used if `data` does not implicitly store these
    :param sample_description: optional pandas.DataFrame containing sample annotations
    """
    gene_names = _parse_gene_names(data, gene_names)
    grouping = _parse_grouping(data, grouping, sample_description)
    x0, x1 = _split_X(data, grouping)
    
    pval = stats.wilcoxon(x0=x0.data, x1=x1.data)
    logfc = np.log(np.mean(x1, axis=0)) - np.log(np.mean(x0, axis=0)).data
    
    de_test = DifferentialExpressionTestWilcoxon(
        gene_ids=gene_names,
        pval=pval,
        logfc=logfc,
    )
    
    return de_test


def two_sample(
        data,
        grouping: Union[str, np.ndarray, list],
        test=None,
        gene_names=None,
        sample_description=None,
        noise_model: str = None,
        batch_size: int = None,
        training_strategy: Union[str, List[Dict[str, object]], Callable] = "AUTO",
) -> _DifferentialExpressionTestSingle:
    """
    Perform differential expression test between two groups on adata object
    for each gene.

    This function wraps the selected statistical test for the scenario of
    a two sample comparison. All unit_test offered in this wrapper
    test for the difference of the mean parameter of both samples.

    The exact unit_test are as follows (assuming the group labels
    are saved in a column named "group"):
    - lrt(log-likelihood ratio test):
        Requires the fitting of 2 generalized linear models (full and reduced).
        
        * full model location parameter: ~ 1 + group
        * full model scale parameter: ~ 1 + group
        * reduced model location parameter: ~ 1
        * reduced model scale parameter: ~ 1 + group
    - Wald test:
        Requires the fitting of 1 generalized linear models.
        model location parameter: ~ 1 + group
        model scale parameter: ~ 1 + group
        Test the group coefficient of the location parameter model against 0.
    - t-test:
        Doesn't require fitting of generalized linear models.
        Welch's t-test between both observation groups.
    - wilcoxon:
        Doesn't require fitting of generalized linear models.
        Wilcoxon rank sum (Mann-Whitney U) test between both observation groups.
        
    :param data: input data
    :param grouping: str, array
    
        - column in data.obs/sample_description which contains the split of observations into the two groups.
        - array of length `num_observations` containing group labels
    :param test: str, statistical test to use. Possible options:
    
        - 'wald': default
        - 'lrt'
        - 't-test'
        - 'wilcoxon'
    :param gene_names: optional list/array of gene names which will be used if `data` does not implicitly store these
    :param sample_description: optional pandas.DataFrame containing sample annotations
    :param noise_model: str, noise model to use in model-based unit_test. Possible options:
        
        - 'nb': default
    :param batch_size: the batch size to use for the estimator
    :param training_strategy: {str, function, list} training strategy to use. Can be:

        - str: will use Estimator.TrainingStrategy[training_strategy] to train
        - function: Can be used to implement custom training function will be called as
          `training_strategy(estimator)`.
        - list of keyword dicts containing method arguments: Will call Estimator.train() once with each dict of
          method arguments.
          
          Example:
          
          .. code-block:: python
          
              [
                {"learning_rate": 0.5, },
                {"learning_rate": 0.05, },
              ]

          This will run training first with learning rate = 0.5 and then with learning rate = 0.05.
    """
    if test in ['t-test', 'wilcoxon'] and noise_model is not None:
        raise ValueError('base.two_sample(): Do not specify `noise_model` if using test t-test or wilcoxon: ' +
                         'The t-test is based on a gaussian noise model and wilcoxon is model free.')
    
    X = _parse_data(data, gene_names)
    gene_names = _parse_gene_names(data, gene_names)
    grouping = _parse_grouping(data, sample_description, grouping)
    sample_description = pd.DataFrame({"grouping": grouping})
    
    groups = np.unique(grouping)
    if groups.size > 2:
        raise ValueError("More than two groups detected:\n\t%s", groups)
    if groups.size < 2:
        raise ValueError("Less than two groups detected:\n\t%s", groups)
    
    # Set default test:
    if test is None:
        test = 'wald'
    
    if test == 'wald':
        if noise_model is None:
            raise ValueError("Please specify noise_model")
        formula_loc = '~ 1 + grouping'
        formula_scale = '~ 1 + grouping'
        de_test = test_wald_loc(
            data=X,
            factor_loc_totest="grouping",
            coef_to_test=None,
            formula_loc=formula_loc,
            formula_scale=formula_scale,
            gene_names=gene_names,
            sample_description=sample_description,
            noise_model=noise_model,
            batch_size=batch_size,
            training_strategy=training_strategy,
        )
    elif test == 'lrt':
        if noise_model is None:
            raise ValueError("Please specify noise_model")
        full_formula_loc = '~ 1 + grouping'
        full_formula_scale = '~ 1 + grouping'
        reduced_formula_loc = '~ 1'
        reduced_formula_scale = '~ 1 + grouping'
        de_test = test_lrt(
            data=X,
            full_formula_loc=full_formula_loc,
            reduced_formula_loc=reduced_formula_loc,
            full_formula_scale=full_formula_scale,
            reduced_formula_scale=reduced_formula_scale,
            gene_names=gene_names,
            sample_description=sample_description,
            noise_model=noise_model,
            batch_size=batch_size,
            training_strategy=training_strategy,
        )
    elif test == 't-test':
        de_test = test_t_test(
            data=X,
            gene_names=gene_names,
            grouping=grouping,
        )
    elif test == 'wilcoxon':
        de_test = test_wilcoxon(
            data=X,
            gene_names=gene_names,
            grouping=grouping,
        )
    else:
        raise ValueError('base.two_sample(): Parameter `test` not recognized.')
    
    return de_test


def test_pairwise(
        data,
        grouping: Union[str, np.ndarray, list],
        test: str = 'z-test',
        gene_names: str = None,
        sample_description: pd.DataFrame = None,
        noise_model: str = None,
        batch_size: int = None,
        training_strategy: Union[str, List[Dict[str, object]], Callable] = "AUTO",
        return_full_test_objs: bool = False,
):
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
    
    - lrt(log-likelihood ratio test):
        Requires the fitting of 2 generalized linear models (full and reduced).
        
        * full model location parameter: ~ 1 + group
        * full model scale parameter: ~ 1 + group
        * reduced model location parameter: ~ 1
        * reduced model scale parameter: ~ 1 + group
    - Wald test:
        Requires the fitting of 1 generalized linear models.
        model location parameter: ~ 1 + group
        model scale parameter: ~ 1 + group
        Test the group coefficient of the location parameter model against 0.
    - t-test:
        Doesn't require fitting of generalized linear models.
        Welch's t-test between both observation groups.
    - wilcoxon:
        Doesn't require fitting of generalized linear models.
        Wilcoxon rank sum (Mann-Whitney U) test between both observation groups.
        
    :param data: input data
    :param grouping: str, array
    
        - column in data.obs/sample_description which contains the split of observations into the two groups.
        - array of length `num_observations` containing group labels
    :param test: str, statistical test to use. Possible options:
    
        - 'z-test': default
        - 'wald'
        - 'lrt'
        - 't-test'
        - 'wilcoxon'
    :param gene_names: optional list/array of gene names which will be used if `data` does not implicitly store these
    :param sample_description: optional pandas.DataFrame containing sample annotations
    :param noise_model: str, noise model to use in model-based unit_test. Possible options:
        
        - 'nb': default
    :param batch_size: the batch size to use for the estimator
    :param training_strategy: {str, function, list} training strategy to use. Can be:

        - str: will use Estimator.TrainingStrategy[training_strategy] to train
        - function: Can be used to implement custom training function will be called as
          `training_strategy(estimator)`.
        - list of keyword dicts containing method arguments: Will call Estimator.train() once with each dict of
          method arguments.
          
          Example:
          
          .. code-block:: python
          
              [
                {"learning_rate": 0.5, },
                {"learning_rate": 0.05, },
              ]

          This will run training first with learning rate = 0.5 and then with learning rate = 0.05.
    :param return_full_test_objs: [Debugging] return matrix of test objects; currently valid for test != "z-test"
    """
    # Do not store all models but only p-value and q-value matrix:
    # genes x groups x groups
    X = _parse_data(data, gene_names)
    gene_names = _parse_gene_names(data, gene_names)
    sample_description = _parse_sample_description(data, sample_description)
    grouping = _parse_grouping(data, sample_description, grouping)
    sample_description = pd.DataFrame({"grouping": grouping})
    
    groups = np.unique(grouping)
    pvals = np.tile(np.NaN, [len(groups), len(groups), X.shape[1]])
    pvals[np.eye(pvals.shape[0]).astype(bool)] = 0
    logfc = np.tile(np.NaN, [len(groups), len(groups), X.shape[1]])
    logfc[np.eye(logfc.shape[0]).astype(bool)] = 0
    tests = np.tile([None], [X.shape[1], len(groups), len(groups)])
    
    if test == 'z-test':
        # fit each group individually
        group_models = []
        for g in groups:
            sel = grouping == g
            model = _fit(
                noise_model=noise_model,
                data=X[sel],
                design_loc=np.ones([np.sum(sel), 1]),
                design_scale=np.ones([np.sum(sel), 1]),
                gene_names=gene_names,
                batch_size=batch_size,
                training_strategy=training_strategy,
            )
            group_models.append(model)
        
        # values of parameter estimates: genes x coefficient array with one coefficient per group
        theta_mle = [np.squeeze(np.asarray(e.par_link_loc)) for e in group_models]
        # standard deviation of estimates: genes x coefficient array with one coefficient per group
        theta_sd = [np.asarray(e.fisher_loc) for e in group_models]
        
        for i, g1 in enumerate(groups):
            for j, g2 in enumerate(groups[(i + 1):]):
                j = j + i + 1
                
                pvals[i, j] = stats.two_coef_z_test(theta_mle0=theta_mle[i], theta_mle1=theta_mle[j],
                                                    theta_sd0=theta_sd[i], theta_sd1=theta_sd[j])
                pvals[j, i] = pvals[i, j]
                logfc[i, j] = theta_mle[j] - theta_mle[i]
                logfc[j, i] = logfc[i, j]
    else:
        for i, g1 in enumerate(groups):
            for j, g2 in enumerate(groups[(i + 1):]):
                j = j + i + 1
                
                sel = (grouping == g1) | (grouping == g2)
                de_test_temp = two_sample(
                    data=X[sel],
                    grouping=grouping[sel],
                    test=test,
                    gene_names=gene_names,
                    sample_description=sample_description.iloc[sel],
                    noise_model=noise_model,
                    batch_size=batch_size,
                    training_strategy=training_strategy,
                )
                pvals[i, j] = de_test_temp.pval
                pvals[j, i] = pvals[i, j]
                logfc[i, j] = de_test_temp.log_fold_change()
                logfc[j, i] = - logfc[i, j]
                if return_full_test_objs:
                    tests[i, j] = de_test_temp
                    tests[j, i] = de_test_temp
    
    de_test = DifferentialExpressionTestPairwise(gene_ids=gene_names, pval=pvals, logfc=logfc)
    
    if return_full_test_objs:
        return de_test, tests
    else:
        return de_test


def test_vsrest(
        data,
        grouping: Union[str, np.ndarray, list],
        test: str = 'fast-wald',
        gene_names: str = None,
        sample_description: pd.DataFrame = None,
        noise_model: str = None,
        batch_size: int = None,
        training_strategy: Union[str, List[Dict[str, object]], Callable] = "AUTO",
):
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
    
    - lrt(log-likelihood ratio test):
        Requires the fitting of 2 generalized linear models (full and reduced).
        
        * full model location parameter: ~ 1 + group
        * full model scale parameter: ~ 1 + group
        * reduced model location parameter: ~ 1
        * reduced model scale parameter: ~ 1 + group
    - Wald test:
        Requires the fitting of 1 generalized linear models.
        model location parameter: ~ 1 + group
        model scale parameter: ~ 1 + group
        Test the group coefficient of the location parameter model against 0.
    - t-test:
        Doesn't require fitting of generalized linear models.
        Welch's t-test between both observation groups.
    - wilcoxon:
        Doesn't require fitting of generalized linear models.
        Wilcoxon rank sum (Mann-Whitney U) test between both observation groups.
    - fast-wald:
        Requires the fitting of 1 generalized linear models.
        model location parameter: ~ group
        model scale parameter: ~ group
        Test each groups location coefficient against the overall average expression.
        
    :param data: input data
    :param grouping: str, array
    
        - column in data.obs/sample_description which contains the split of observations into the two groups.
        - array of length `num_observations` containing group labels
    :param test: str, statistical test to use. Possible options:
    
        - 'fast-wald': default
        - 'wald'
        - 'lrt'
        - 't-test'
        - 'wilcoxon'
    :param gene_names: optional list/array of gene names which will be used if `data` does not implicitly store these
    :param sample_description: optional pandas.DataFrame containing sample annotations
    :param noise_model: str, noise model to use in model-based unit_test. Possible options:
        
        - 'nb': default
    :param batch_size: the batch size to use for the estimator
    :param training_strategy: {str, function, list} training strategy to use. Can be:

        - str: will use Estimator.TrainingStrategy[training_strategy] to train
        - function: Can be used to implement custom training function will be called as
          `training_strategy(estimator)`.
        - list of keyword dicts containing method arguments: Will call Estimator.train() once with each dict of
          method arguments.
          
          Example:
          
          .. code-block:: python
          
              [
                {"learning_rate": 0.5, },
                {"learning_rate": 0.05, },
              ]

          This will run training first with learning rate = 0.5 and then with learning rate = 0.05.
    :param return_full_test_objs: [Debugging] return matrix of test objects; currently valid for test != "z-test"
    """
    # Do not store all models but only p-value and q-value matrix:
    # genes x groups
    X = _parse_data(data, gene_names)
    gene_names = _parse_gene_names(data, gene_names)
    sample_description = _parse_sample_description(data, sample_description)
    grouping = _parse_grouping(data, sample_description, grouping)
    sample_description = pd.DataFrame({"grouping": grouping})
    
    groups = np.unique(grouping)
    pvals = np.zeros([len(groups), X.shape[1]])
    logfc = np.zeros([len(groups), X.shape[1]])
    
    if test == 'fast-wald':
        # fit each group individually
        group_models = []
        for g in groups:
            sel = grouping == g
            model = _fit(
                noise_model=noise_model,
                data=X[sel],
                design_loc=np.ones([np.sum(sel), 1]),
                design_scale=np.ones([np.sum(sel), 1]),
                gene_names=gene_names,
                batch_size=batch_size,
                training_strategy=training_strategy,
            )
            group_models.append(model)
        
        # values of parameter estimates: genes x coefficient array with one coefficient per group
        theta_mle = [np.squeeze(np.asarray(e.par_link_loc)) for e in group_models]
        # standard deviation of estimates: genes x coefficient array with one coefficient per group
        # $\text{SE}(\hat{\theta}_{ML}) = \frac{1}{Fisher(\hat{\theta}_{ML})}$
        theta_sd = [1 / np.sqrt(np.asarray(e.fisher_loc)) for e in group_models]
        # average expression
        ave_expr = np.mean(X, axis=0)
        
        for i, g1 in enumerate(groups):
            pvals[i] = stats.wald_test(theta_mle=theta_mle[i], theta_sd=theta_sd[i], theta0=ave_expr)
            logfc[i] = theta_mle[i]
    else:
        for i, g1 in enumerate(groups):
            test_grouping = np.where(grouping == g1, "group", "rest")
            de_test_temp = two_sample(
                data=X,
                grouping=test_grouping,
                test=test,
                gene_names=gene_names,
                sample_description=sample_description,
                noise_model=noise_model,
                batch_size=batch_size,
                training_strategy=training_strategy,
            )
            pvals[i] = de_test_temp.pval
            logfc[i] = de_test_temp.log_fold_change()
    
    de_test = DifferentialExpressionTestVsRest(gene_ids=gene_names, pval=pvals, logfc=logfc)
    return de_test
