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

from ..stats import stats
from . import correction
from ..models.batch_bfgs.optim import Estim_BFGS

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
    def X(self) -> np.ndarray:
        pass

    @property
    @abc.abstractmethod
    def design_loc(self) -> np.ndarray:
        pass

    @property
    @abc.abstractmethod
    def design_scale(self) -> np.ndarray:
        pass

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
    def hessians(self, **kwargs) -> np.ndarray:
        pass

    @property
    @abc.abstractmethod
    def fisher_inv(self, **kwargs) -> np.ndarray:
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
        self._mean = None

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

    def _ave(self):
        """
        Returns a xr.DataArray containing the mean expression by gene

        :return: xr.DataArray
        """
        pass

    @property
    def mean(self):
        if self._mean is None:
            self._mean = self._ave()
        return self._mean

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

    def plot_volcano(self):
        """
        returns a volcano plot of p-value vs. log fold change

        :return: Tuple of matplotlib (figure, axis)
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        pvals = np.reshape(self.pval, -1)
        pvals = np.nextafter(0, 1, out=pvals, where=pvals == 0)
        neg_log_pvals = -(np.log(pvals) / np.log(10))
        neg_log_pvals = np.clip(neg_log_pvals, 0, 30, neg_log_pvals)
        logfc = np.reshape(self.log2_fold_change(), -1)

        fig, ax = plt.subplots()

        sns.scatterplot(y=neg_log_pvals, x=logfc, ax=ax)

        ax.set(xlabel="log2FC", ylabel='-log10(pval)')

        return fig, ax

    def plot_diagnostics(self):
        """
        Directly plots a set of diagnostic diagrams
        """
        import matplotlib.pyplot as plt

        volcano = self.plot_volcano()
        plt.show()


class _DifferentialExpressionTestSingle(_DifferentialExpressionTest, metaclass=abc.ABCMeta):
    """
    _DifferentialExpressionTest for unit_test with a single test per gene.
    The individual test object inherit directly from this class.

    All implementations of this class should return one p-value and one fold change per gene.
    """

    def _threshold_summary(self, res, qval_thres=None, 
        log2fc_upper_thres=None, log2fc_lower_thres=None, mean_thres=None) -> pd.DataFrame:
        """
        Reduce differential expression results into an output table with desired thresholds.
        """
        if qval_thres is not None:
            res = res.iloc[res['qval'].values <= qval_thres,:]

        if log2fc_upper_thres is not None and log2fc_lower_thres is None:
            res = res.iloc[res['log2fc'].values >= log2fc_upper_thres,:]
        elif log2fc_upper_thres is None and log2fc_lower_thres is not None:
            res = res.iloc[res['log2fc'].values <= log2fc_lower_thres,:]
        elif log2fc_upper_thres is not None and log2fc_lower_thres is not None:
            res = res.iloc[np.logical_or(
                res['log2fc'].values <= log2fc_lower_thres,
                res['log2fc'].values >= log2fc_upper_thres),:]

        if mean_thres is not None:
            res = res.iloc[res['mean'].values >= mean_thres,:]

        return res

    def summary(self, qval_thres=None, 
        log2fc_upper_thres=None, log2fc_lower_thres=None, mean_thres=None, **kwargs) -> pd.DataFrame:
        """
        Summarize differential expression results into an output table.
        """
        assert self.gene_ids is not None

        res = pd.DataFrame({
            "gene": self.gene_ids,
            "pval": self.pval,
            "qval": self.qval,
            "log2fc": self.log2_fold_change(),
            "mean": self.mean
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

        if np.any(full < reduced):
            logger.warning("Test assumption failed: full model is (partially) less probable than reduced model!")

        return stats.likelihood_ratio_test(
            ll_full=full,
            ll_reduced=reduced,
            df_full=self.full_estim.design_loc.shape[-1] + self.full_estim.design_scale.shape[-1],
            df_reduced=self.reduced_estim.design_loc.shape[-1] + self.reduced_estim.design_scale.shape[-1],
        )

    def _ave(self):
        """
        Returns a xr.DataArray containing the mean expression by gene

        :return: xr.DataArray
        """

        return np.mean(self.full_estim.X, axis=0)

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

    def summary(self, qval_thres=None, log2fc_upper_thres=None, 
        log2fc_lower_thres=None, mean_thres=None, 
        **kwargs) -> pd.DataFrame:
        """
        Summarize differential expression results into an output table.
        """
        res = super().summary(**kwargs)
        res["grad"] = self.full_model_gradient.data
        res["grad_red"] = self.reduced_model_gradient.data

        res = self._threshold_summary(
            res=res, 
            qval_thres=qval_thres, 
            log2fc_upper_thres=log2fc_upper_thres, 
            log2fc_lower_thres=log2fc_lower_thres, 
            mean_thres=mean_thres
            )

        return res


class DifferentialExpressionTestWald(_DifferentialExpressionTestSingle):
    """
    Single wald test per gene.
    """

    model_estim: _Estimation
    coef_loc_totest: int
    theta_mle: np.ndarray
    theta_sd: np.ndarray

    def __init__(self, model_estim: _Estimation, col_index):
        super().__init__()
        self.model_estim = model_estim
        self.coef_loc_totest = col_index
        # p = self.pval
        # q = self.qval

        try:
            if model_estim._error_codes is not None:
                self._error_codes = model_estim._error_codes
        except Exception as e:
            self._error_codes = None

        try:
            if model_estim._niter is not None:
                self._niter = model_estim._niter
        except Exception as e:
            self._niter = None

    @property
    def log_probs(self):
        return np.sum(self.model_estim.log_probs(), axis=0)

    @property
    def gene_ids(self) -> np.ndarray:
        return np.asarray(self.model_estim.features)

    @property
    def model_gradient(self):
        return self.model_estim.gradient

    def _ave(self):
        """
        Returns a xr.DataArray containing the mean expression by gene

        :return: xr.DataArray
        """

        return np.mean(self.model_estim.X, axis=0)

    def log_fold_change(self, base=np.e, **kwargs):
        """
        Returns one fold change per gene
        """
        # design = np.unique(self.model_estim.design_loc, axis=0)
        # dmat = np.zeros_like(design)
        # dmat[:, self.coef_loc_totest] = design[:, self.coef_loc_totest]

        # loc = dmat @ self.model_estim.par_link_loc[self.coef_loc_totest]
        # return loc[1] - loc[0]
        return self.model_estim.par_link_loc[self.coef_loc_totest]

    def _test(self):
        self.theta_mle = self.model_estim.par_link_loc[self.coef_loc_totest]
        # standard deviation of estimates: coefficients x genes array with one coefficient per group
        # theta_sd = sqrt(diagonal(fisher_inv))
        self.theta_sd = np.sqrt(np.diagonal(self.model_estim.fisher_inv, axis1=-2, axis2=-1)).T[self.coef_loc_totest]

        return stats.wald_test(theta_mle=self.theta_mle, theta_sd=self.theta_sd, theta0=0)

    def summary(self, qval_thres=None, log2fc_upper_thres=None, 
        log2fc_lower_thres=None, mean_thres=None, 
        **kwargs) -> pd.DataFrame:
        """
        Summarize differential expression results into an output table.
        """
        res = super().summary(**kwargs)
        res["grad"] = self.model_gradient.data
        res["coef_mle"] = self.theta_mle
        res["coef_sd"] = self.theta_sd
        # add in info from bfgs
        if self.log_probs is not None:
            res["ll"] = self.log_probs
        if self._error_codes is not None:
            res["err"] = self._error_codes
        if self._niter is not None:
            res["niter"] = self._niter

        res = self._threshold_summary(
            res=res, 
            qval_thres=qval_thres, 
            log2fc_upper_thres=log2fc_upper_thres, 
            log2fc_lower_thres=log2fc_lower_thres, 
            mean_thres=mean_thres
            )

        return res

    def plot_vs_ttest(self):
        import matplotlib.pyplot as plt
        import seaborn as sns

        grouping = np.asarray(self.model_estim.design_loc[:, self.coef_loc_totest])
        ttest = t_test(
            data=self.model_estim.X,
            grouping=grouping,
            gene_ids=self.gene_ids,
        )
        ttest_pvals = ttest.pval

        fig, ax = plt.subplots()

        sns.scatterplot(x=ttest_pvals, y=self.pval, ax=ax)

        ax.set(xlabel="t-test", ylabel='wald test')

        return fig, ax

    def plot_diagnostics(self):
        import matplotlib.pyplot as plt

        volcano = self.plot_volcano()
        plt.show()
        ttest_comp = self.plot_vs_ttest()
        plt.show()


class DifferentialExpressionTestTT(_DifferentialExpressionTestSingle):
    """
    Single t-test test per gene.
    """

    def __init__(self, data, grouping, gene_ids):
        super().__init__()
        self.data = data
        self.grouping = grouping
        self._gene_ids = np.asarray(gene_ids)

        x0, x1 = _split_X(data, grouping)

        # Only compute p-values for genes with non-zero observations and non-zero group-wise variance.
        self._mean = np.mean(data, axis=0)
        self._ave_geq_zero = np.asarray(self.mean).flatten() > 0
        self._var_geq_zero = np.logical_or(
            np.asarray(np.var(x0, axis=0)).flatten() > 0,
            np.asarray(np.var(x1, axis=0)).flatten() > 0
        )
        idx_tt = np.where(np.logical_and(self._ave_geq_zero == True, self._var_geq_zero == True))[0]
        pval = np.zeros([self._gene_ids.shape[0]]) + np.nan
        pval[idx_tt] = stats.t_test_raw(x0=x0[:, idx_tt], x1=x1[:, idx_tt])
        self._pval = pval
        self._logfc = np.log(np.mean(x1, axis=0)) - np.log(np.mean(x0, axis=0)).data
        # Return 0 if LFC was non-zero and variances are zero,
        # this causes division by zero in the test statistic. This
        # is a highly significant result if one believes the variance estimate.
        pval[np.logical_and(np.logical_and(self._var_geq_zero == False, self._ave_geq_zero == True),
                            self._logfc != 0)] = 0
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

    def summary(self, qval_thres=None, log2fc_upper_thres=None, 
        log2fc_lower_thres=None, mean_thres=None, 
        **kwargs) -> pd.DataFrame:
        """
        Summarize differential expression results into an output table.
        """
        res = super().summary(**kwargs)
        res["zero_mean"] = self._ave_geq_zero == False
        res["zero_variance"] = self._var_geq_zero == False

        res = self._threshold_summary(
            res=res, 
            qval_thres=qval_thres, 
            log2fc_upper_thres=log2fc_upper_thres, 
            log2fc_lower_thres=log2fc_lower_thres, 
            mean_thres=mean_thres
            )

        return res


class DifferentialExpressionTestWilcoxon(_DifferentialExpressionTestSingle):
    """
    Single wilcoxon rank sum test per gene.
    """

    def __init__(self, data, grouping, gene_ids):
        super().__init__()
        self.data = data
        self.grouping = grouping
        self._gene_ids = np.asarray(gene_ids)

        x0, x1 = _split_X(data, grouping)

        self._mean = np.mean(data, axis=0)
        self._pval = stats.wilcoxon_test(x0=x0.data, x1=x1.data)
        self._logfc = np.log(np.mean(x1, axis=0)) - np.log(np.mean(x0, axis=0)).data
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

    def summary(self, qval_thres=None, log2fc_upper_thres=None, 
        log2fc_lower_thres=None, mean_thres=None, 
        **kwargs) -> pd.DataFrame:
        """
        Summarize differential expression results into an output table.
        """
        res = super().summary(**kwargs)
        
        res = self._threshold_summary(
            res=res, 
            qval_thres=qval_thres, 
            log2fc_upper_thres=log2fc_upper_thres, 
            log2fc_lower_thres=log2fc_lower_thres, 
            mean_thres=mean_thres
            )

        return res

    def plot_vs_ttest(self):
        import matplotlib.pyplot as plt
        import seaborn as sns

        grouping = self.grouping
        ttest = t_test(
            data=self.data,
            grouping=grouping,
            gene_names=self.gene_ids,
        )
        ttest_pvals = ttest.pval

        fig, ax = plt.subplots()

        sns.scatterplot(x=ttest_pvals, y=self.pval, ax=ax)

        ax.set(xlabel="t-test", ylabel='wilcoxon test')

        return fig, ax

    def plot_diagnostics(self):
        import matplotlib.pyplot as plt

        volcano = self.plot_volcano()
        plt.show()
        ttest_comp = self.plot_vs_ttest()
        plt.show()


class _DifferentialExpressionTestMulti(_DifferentialExpressionTest, metaclass=abc.ABCMeta):
    """
    _DifferentialExpressionTest for unit_test with a multiple unit_test per gene.
    The individual test object inherit directly from this class.
    """

    def __init__(self, correction_type: str):
        """

        :param correction_type: Choose between global and test-wise correction.
            Can be:

            - "global": correct all p-values in one operation
            - "by_test": correct the p-values of each test individually
        """
        super().__init__()
        self._correction_type = correction_type

    def _correction(self, method):
        if self._correction_type.lower() == "global":
            pvals = np.reshape(self.pval, -1)
            qvals = correction.correct(pvals=pvals, method=method)
            qvals = np.reshape(qvals, self.pval.shape)
            return qvals
        elif self._correction_type.lower() == "by_test":
            qvals = np.apply_along_axis(
                func1d=lambda pvals: correction.correct(pvals=pvals, method=method),
                axis=-1,
                arr=self.pval,
            )
            return qvals

    def summary(self, **kwargs) -> pd.DataFrame:
        """
        Summarize differential expression results into an output table.

        :return: pandas.DataFrame with the following columns:

            - gene: the gene id's
            - pval: the minimum per-gene p-value of all tests
            - qval: the minimum per-gene q-value of all tests
            - log2fc: the maximal/minimal (depending on which one is higher) log2 fold change of the genes
            - mean: the mean expression of the gene across all groups
        """
        assert self.gene_ids is not None

        # calculate maximum logFC of lower triangular fold change matrix
        raw_logfc = self.log2_fold_change()

        # first flatten all dimensions up to the last 'gene' dimension
        flat_logfc = raw_logfc.reshape(-1, raw_logfc.shape[-1])
        # next, get argmax of flattened logfc and unravel the true indices from it
        r, c = np.unravel_index(flat_logfc.argmax(0), raw_logfc.shape[:2])
        # if logfc is maximal in the lower triangular matrix, multiply it with -1
        logfc = raw_logfc[r, c, np.arange(raw_logfc.shape[-1])] * np.where(r <= c, 1, -1)

        res = pd.DataFrame({
            "gene": self.gene_ids,
            # return minimal pval by gene:
            "pval": np.min(self.pval.reshape(-1, self.pval.shape[-1]), axis=0),
            # return minimal qval by gene:
            "qval": np.min(self.qval.reshape(-1, self.qval.shape[-1]), axis=0),
            # return maximal logFC by gene:
            "log2fc": np.asarray(logfc),
            # return mean expression across all groups by gene:
            "mean": np.asarray(self.mean)
        })

        return res


class DifferentialExpressionTestPairwise(_DifferentialExpressionTestMulti):
    """
    Pairwise unit_test between more than 2 groups per gene.
    """

    def __init__(self, gene_ids, pval, logfc, ave, groups, tests, correction_type: str):
        super().__init__(correction_type=correction_type)
        self._gene_ids = np.asarray(gene_ids)
        self._logfc = logfc
        self._pval = pval
        self._mean = ave
        self.groups = list(np.asarray(groups))
        self._tests = tests
        
        q = self.qval

    @property
    def gene_ids(self) -> np.ndarray:
        return self._gene_ids

    @property
    def tests(self):
        """
        If `keep_full_test_objs` was set to `True`, this will return a matrix of differential expression tests.
        """
        if self._tests is None:
            raise ValueError("Individual tests were not kept!")

        return self._tests

    def log_fold_change(self, base=np.e, **kwargs):
        """
        Returns matrix of fold changes per gene
        """
        if base == np.e:
            return self._logfc
        else:
            return self._logfc / np.log(base)

    def _check_groups(self, group1, group2):
        if group1 not in self.groups:
            raise ValueError('group1 not recognized')
        if group2 not in self.groups:
            raise ValueError('group2 not recognized')

    def pval_pair(self, group1, group2):
        assert self._pval is not None

        self._check_groups(group1, group2)
        return self._pval[self.groups.index(group1),self.groups.index(group2),:]

    def qval_pair(self, group1, group2):
        assert self._qval is not None

        self._check_groups(group1, group2)
        return self._qval[self.groups.index(group1),self.groups.index(group2),:]

    def log_fold_change_pair(self, group1, group2, base=np.e):
        assert self._logfc is not None

        self._check_groups(group1, group2)
        return self.log_fold_change(base=base)[self.groups.index(group1),self.groups.index(group2),:]

    def summary_pair(self, group1, group2, **kwargs) -> pd.DataFrame:
        """
        Summarize differential expression results into an output table.

        :return: pandas.DataFrame with the following columns:

            - gene: the gene id's
            - pval: the per-gene p-value of the selected test
            - qval: the per-gene q-value of the selected test
            - log2fc: the per-gene log2 fold change of the selected test
            - mean: the mean expression of the gene across all groups
        """
        assert self.gene_ids is not None

        res = pd.DataFrame({
            "gene": self.gene_ids,
            "pval": self.pval_pair(group1=group1, group2=group2),
            "qval": self.qval_pair(group1=group1, group2=group2),
            "log2fc": self.log_fold_change_pair(group1=group1, group2=group2, base=2),
            "mean": np.asarray(self.mean)
        })

        return res


class DifferentialExpressionTestZTest(_DifferentialExpressionTestMulti):
    """
    Pairwise unit_test between more than 2 groups per gene.
    """

    model_estim: _Estimation
    theta_mle: np.ndarray
    theta_sd: np.ndarray

    def __init__(self, model_estim: _Estimation, grouping, groups, correction_type: str):
        super().__init__(correction_type=correction_type)
        self.model_estim = model_estim
        self.grouping = grouping
        self.groups = list(np.asarray(groups))

        # values of parameter estimates: coefficients x genes array with one coefficient per group
        self._theta_mle = model_estim.par_link_loc
        # standard deviation of estimates: coefficients x genes array with one coefficient per group
        # theta_sd = sqrt(diagonal(fisher_inv))
        self._theta_sd = np.sqrt(np.diagonal(model_estim.fisher_inv, axis1=-2, axis2=-1)).T
        self._logfc = None

        # Call tests in constructor.
        p = self.pval
        q = self.qval

    def _test(self, **kwargs):
        groups = self.groups
        num_features = self.model_estim.X.shape[1]

        pvals = np.tile(np.NaN, [len(groups), len(groups), num_features])
        pvals[np.eye(pvals.shape[0]).astype(bool)] = 1

        theta_mle = self._theta_mle
        theta_sd = self._theta_sd

        for i, g1 in enumerate(groups):
            for j, g2 in enumerate(groups[(i + 1):]):
                j = j + i + 1

                pvals[i, j] = stats.two_coef_z_test(theta_mle0=theta_mle[i], theta_mle1=theta_mle[j],
                                                    theta_sd0=theta_sd[i], theta_sd1=theta_sd[j])
                pvals[j, i] = pvals[i, j]

        return pvals

    @property
    def log_probs(self):
        return np.sum(self.model_estim.log_probs(), axis=0)

    @property
    def gene_ids(self) -> np.ndarray:
        return np.asarray(self.model_estim.features)

    @property
    def model_gradient(self):
        return self.model_estim.gradient

    def _ave(self):
        """
        Returns a xr.DataArray containing the mean expression by gene

        :return: xr.DataArray
        """

        return np.mean(self.model_estim.X, axis=0)

    def log_fold_change(self, base=np.e, **kwargs):
        """
        Returns matrix of fold changes per gene
        """
        if self._logfc is None:
            groups = self.groups
            num_features = self.model_estim.X.shape[1]

            logfc = np.tile(np.NaN, [len(groups), len(groups), num_features])
            logfc[np.eye(logfc.shape[0]).astype(bool)] = 0

            theta_mle = self._theta_mle

            for i, g1 in enumerate(groups):
                for j, g2 in enumerate(groups[(i + 1):]):
                    j = j + i + 1

                    logfc[i, j] = theta_mle[j] - theta_mle[i]
                    logfc[j, i] = -logfc[i, j]

            self._logfc = logfc

        if base == np.e:
            return self._logfc
        else:
            return self._logfc / np.log(base)

    def _check_groups(self, group1, group2):
        if group1 not in self.groups:
            raise ValueError('group1 not recognized')
        if group2 not in self.groups:
            raise ValueError('group2 not recognized')

    def pval_pair(self, group1, group2):
        self._check_groups(group1, group2)
        return self.pval[self.groups.index(group1),self.groups.index(group2),:]

    def qval_pair(self, group1, group2):
        self._check_groups(group1, group2)
        return self.qval[self.groups.index(group1),self.groups.index(group2),:]

    def log_fold_change_pair(self, group1, group2, base=np.e):
        self._check_groups(group1, group2)
        return self.log_fold_change(base=base)[self.groups.index(group1),self.groups.index(group2),:]

    def summary_pair(self, group1, group2, **kwargs) -> pd.DataFrame:
        """
        Summarize differential expression results into an output table.

        :return: pandas.DataFrame with the following columns:

            - gene: the gene id's
            - pval: the per-gene p-value of the selected test
            - qval: the per-gene q-value of the selected test
            - log2fc: the per-gene log2 fold change of the selected test
            - mean: the mean expression of the gene across all groups
        """
        assert self.gene_ids is not None

        res = pd.DataFrame({
            "gene": self.gene_ids,
            "pval": self.pval_pair(group1=group1, group2=group2),
            "qval": self.qval_pair(group1=group1, group2=group2),
            "log2fc": self.log_fold_change_pair(group1=group1, group2=group2, base=2),
            "mean": np.asarray(self.mean)
        })

        return res


class DifferentialExpressionTestVsRest(_DifferentialExpressionTestMulti):
    """
    Tests between between each group and the rest for more than 2 groups per gene.
    """

    def __init__(self, gene_ids, pval, logfc, ave, groups, tests, correction_type: str):
        super().__init__(correction_type=correction_type)
        self._gene_ids = np.asarray(gene_ids)
        self._pval = pval
        self._logfc = logfc
        self._mean = ave
        self.groups = list(np.asarray(groups))
        self._tests = tests

        q = self.qval

    @property
    def tests(self):
        """
        If `keep_full_test_objs` was set to `True`, this will return a matrix of differential expression tests.
        """
        if self._tests is None:
            raise ValueError("Individual tests were not kept!")

        return self._tests

    @property
    def gene_ids(self) -> np.ndarray:
        return self._gene_ids

    def log_fold_change(self, base=np.e, **kwargs):
        if base == np.e:
            return self._logfc
        else:
            return self._logfc / np.log(base)

    def _check_group(self, group):
        if group not in self.groups:
            raise ValueError('group not recognized')

    def pval_group(self, group):
        self._check_group(group)
        return self.pval[0,self.groups.index(group),:]

    def qval_group(self, group):
        self._check_group(group)
        return self.qval[0,self.groups.index(group),:]

    def log_fold_change_group(self, group, base=np.e):
        self._check_group(group)
        return self.log_fold_change(base=base)[0,self.groups.index(group),:]

    def summary_group(self, group, **kwargs) -> pd.DataFrame:
        """
        Summarize differential expression results into an output table.

        :return: pandas.DataFrame with the following columns:

            - gene: the gene id's
            - pval: the per-gene p-value of the selected test
            - qval: the per-gene q-value of the selected test
            - log2fc: the per-gene log2 fold change of the selected test
            - mean: the mean expression of the gene across all groups
        """
        assert self.gene_ids is not None

        res = pd.DataFrame({
            "gene": self.gene_ids,
            "pval": self.pval_group(group=group),
            "qval": self.qval_group(group=group),
            "log2fc": self.log_fold_change_group(group=group, base=2),
            "mean": np.asarray(self.mean)
        })

        return res

class DifferentialExpressionTestByPartition(_DifferentialExpressionTestMulti):
    """
    Stores a particular test performed within each partition of the data set.
    """

    def __init__(self, partitions, tests, ave, correction_type: str = "by_test"):
        super().__init__(correction_type=correction_type)
        self.partitions = list(np.asarray(partitions))
        self._tests = tests
        self._gene_ids = tests[0].gene_ids
        self._pval = np.expand_dims(np.vstack([x.pval for x in tests]), axis=0)
        self._logfc = np.expand_dims(np.vstack([x.log_fold_change() for x in tests]), axis=0)
        self._mean = ave

        q = self.qval

    @property
    def gene_ids(self) -> np.ndarray:
        return self._gene_ids

    def log_fold_change(self, base=np.e, **kwargs):
        if base == np.e:
            return self._logfc
        else:
            return self._logfc / np.log(base)

    def _check_partition(self, partition):
        if partition not in self.partitions:
            raise ValueError('partition not recognized')

    @property
    def tests(self, partition=None):
        """
        If `keep_full_test_objs` was set to `True`, this will return a matrix of differential expression tests.

        :param partition: The partition for which to return the test. Returns full list if None.
        """
        if self._tests is None:
            raise ValueError("Individual tests were not kept!")

        if partition is None:
            return self._tests
        else:
            self._check_partition(partition)
            return self._tests[self.partitions.index(partition)]


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
        quick_scale: bool = None,
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
    :param quick_scale: Depending on the optimizer, `scale` will be fitted faster and maybe less accurate.

        Useful in scenarios where fitting the exact `scale` is not absolutely necessary.
    :param close_session: If True, will finalize the estimator. Otherwise, return the estimator itself.
    """
    if training_strategy.lower() == 'bfgs':
        lib_size = np.zeros(data.shape[0])
        if noise_model == "nb" or noise_model == "negative_binomial":
            estim = Estim_BFGS(X=data, design_loc=design_loc, design_scale=design_scale,
                               lib_size=lib_size, batch_size=batch_size, feature_names=gene_names)
            estim.run(nproc=3, maxiter=10000, debug=False)
            model = estim.return_batchglm_formated_model()
        else:
            raise ValueError('base.test(): `noise_model` not recognized.')
    else:
        if noise_model == "nb" or noise_model == "negative_binomial":
            import batchglm.api.models.nb_glm as test_model

            logger.info("Estimating model...")
            input_data = test_model.InputData.new(
                data=data,
                design_loc=design_loc,
                design_scale=design_scale,
                feature_names=gene_names,
            )

            constructor_args = {}
            if batch_size is not None:
                constructor_args["batch_size"] = batch_size
            if quick_scale is not None:
                constructor_args["quick_scale"] = quick_scale,
            estim = test_model.Estimator(
                input_data=input_data,
                init_model=init_model,
                **constructor_args
            )

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


def lrt(
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
        quick_scale: bool = None,
        **kwargs
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
    :param quick_scale: Depending on the optimizer, `scale` will be fitted faster and maybe less accurate.

        Useful in scenarios where fitting the exact `scale` is not absolutely necessary.
    :param kwargs: [Debugging] Additional arguments will be passed to the _fit method.
    """
    if len(kwargs) != 0:
        logger.info("additional kwargs: %s", str(kwargs))

    # TODO: remove this warning when lrt is working
    logger.warning("lrt is not ready for usage yet!")

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
        quick_scale=quick_scale,
        **kwargs,
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
        quick_scale=quick_scale,
        **kwargs,
    )

    de_test = DifferentialExpressionTestLRT(
        sample_description=sample_description,
        full_design_loc_info=full_design_loc.design_info,
        full_estim=full_model,
        reduced_design_loc_info=reduced_design_loc.design_info,
        reduced_estim=reduced_model,
    )

    return de_test


def wald(
        data,
        factor_loc_totest: str,
        coef_to_test: object = None,  # e.g. coef_to_test="B"
        formula: str = None,
        formula_loc: str = None,
        formula_scale: str = None,
        gene_names: Union[str, np.ndarray] = None,
        sample_description: pd.DataFrame = None,
        noise_model: str = "nb",
        batch_size: int = None,
        training_strategy: Union[str, List[Dict[str, object]], Callable] = "AUTO",
        quick_scale: bool = None,
        **kwargs
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
    :param quick_scale: Depending on the optimizer, `scale` will be fitted faster and maybe less accurate.

        Useful in scenarios where fitting the exact `scale` is not absolutely necessary.
    :param kwargs: [Debugging] Additional arguments will be passed to the _fit method.
    """
    if len(kwargs) != 0:
        logger.debug("additional kwargs: %s", str(kwargs))

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
        quick_scale=quick_scale,
        **kwargs,
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


def t_test(
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
    :param gene_ids: optional list/array of gene names which will be used if `data` does not implicitly store these
    :param sample_description: optional pandas.DataFrame containing sample annotations
    """
    gene_ids = _parse_gene_names(data, gene_names)
    X = _parse_data(data, gene_names)
    grouping = _parse_grouping(data, sample_description, grouping)

    de_test = DifferentialExpressionTestTT(
        data=X,
        grouping=grouping,
        gene_ids=gene_names,
    )

    return de_test


def wilcoxon(
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
    grouping = _parse_grouping(data, sample_description, grouping)

    de_test = DifferentialExpressionTestWilcoxon(
        data=data,
        grouping=grouping,
        gene_ids=gene_names,
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
        quick_scale: bool = None,
        **kwargs
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
    :param quick_scale: Depending on the optimizer, `scale` will be fitted faster and maybe less accurate.

        Useful in scenarios where fitting the exact `scale` is not absolutely necessary.
    :param kwargs: [Debugging] Additional arguments will be passed to the _fit method.
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

    if test.lower() == 'wald':
        if noise_model is None:
            raise ValueError("Please specify noise_model")
        formula_loc = '~ 1 + grouping'
        formula_scale = '~ 1 + grouping'
        de_test = wald(
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
            quick_scale=quick_scale,
            **kwargs
        )
    elif test.lower() == 'lrt':
        if noise_model is None:
            raise ValueError("Please specify noise_model")
        full_formula_loc = '~ 1 + grouping'
        full_formula_scale = '~ 1 + grouping'
        reduced_formula_loc = '~ 1'
        reduced_formula_scale = '~ 1 + grouping'
        de_test = lrt(
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
            quick_scale=quick_scale,
            **kwargs
        )
    elif test.lower() == 't-test' or test.lower() == "t_test" or test.lower() == "ttest":
        de_test = t_test(
            data=X,
            gene_names=gene_names,
            grouping=grouping,
        )
    elif test.lower() == 'wilcoxon':
        de_test = wilcoxon(
            data=X,
            gene_names=gene_names,
            grouping=grouping,
        )
    else:
        raise ValueError('base.two_sample(): Parameter `test` not recognized.')

    return de_test


def pairwise(
        data,
        grouping: Union[str, np.ndarray, list],
        test: str = 'z-test',
        gene_names: str = None,
        sample_description: pd.DataFrame = None,
        noise_model: str = None,
        pval_correction: str = "global",
        batch_size: int = None,
        training_strategy: Union[str, List[Dict[str, object]], Callable] = "AUTO",
        quick_scale: bool = None,
        keep_full_test_objs: bool = False,
        **kwargs
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
    :param pval_correction: Choose between global and test-wise correction.
        Can be:

        - "global": correct all p-values in one operation
        - "by_test": correct the p-values of each test individually
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
    :param quick_scale: Depending on the optimizer, `scale` will be fitted faster and maybe less accurate.

        Useful in scenarios where fitting the exact `scale` is not absolutely necessary.
    :param keep_full_test_objs: [Debugging] keep the individual test objects; currently valid for test != "z-test"
    :param kwargs: [Debugging] Additional arguments will be passed to the _fit method.
    """
    if len(kwargs) != 0:
        logger.info("additional kwargs: %s", str(kwargs))

    # Do not store all models but only p-value and q-value matrix:
    # genes x groups x groups
    X = _parse_data(data, gene_names)
    gene_names = _parse_gene_names(data, gene_names)
    sample_description = _parse_sample_description(data, sample_description)
    grouping = _parse_grouping(data, sample_description, grouping)
    sample_description = pd.DataFrame({"grouping": grouping})

    if test.lower() == 'z-test' or test.lower() == 'z_test' or test.lower() == 'ztest':
        # -1 in formula removes intercept
        dmat = data_utils.design_matrix(sample_description, formula="~ 1 - 1 + grouping")
        model = _fit(
            noise_model=noise_model,
            data=X,
            design_loc=dmat,
            design_scale=dmat,
            gene_names=gene_names,
            batch_size=batch_size,
            training_strategy=training_strategy,
            quick_scale=quick_scale,
            **kwargs
        )

        # # values of parameter estimates: coefficients x genes array with one coefficient per group
        # theta_mle = model.par_link_loc
        # # standard deviation of estimates: coefficients x genes array with one coefficient per group
        # # theta_sd = sqrt(diagonal(fisher_inv))
        # theta_sd = np.sqrt(np.diagonal(model.fisher_inv, axis1=-2, axis2=-1)).T
        #
        # for i, g1 in enumerate(groups):
        #     for j, g2 in enumerate(groups[(i + 1):]):
        #         j = j + i + 1
        #
        #         pvals[i, j] = stats.two_coef_z_test(theta_mle0=theta_mle[i], theta_mle1=theta_mle[j],
        #                                             theta_sd0=theta_sd[i], theta_sd1=theta_sd[j])
        #         pvals[j, i] = pvals[i, j]
        #         logfc[i, j] = theta_mle[j] - theta_mle[i]
        #         logfc[j, i] = logfc[i, j]

        de_test = DifferentialExpressionTestZTest(
            model_estim=model,
            grouping=grouping,
            groups=np.unique(grouping),
            correction_type=pval_correction
        )
    else:
        groups = np.unique(grouping)
        pvals = np.tile(np.NaN, [len(groups), len(groups), X.shape[1]])
        pvals[np.eye(pvals.shape[0]).astype(bool)] = 0
        logfc = np.tile(np.NaN, [len(groups), len(groups), X.shape[1]])
        logfc[np.eye(logfc.shape[0]).astype(bool)] = 0

        if keep_full_test_objs:
            tests = np.tile([None], [X.shape[1], len(groups), len(groups)])
        else:
            tests = None

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
                    quick_scale=quick_scale,
                    **kwargs
                )
                pvals[i, j] = de_test_temp.pval
                pvals[j, i] = pvals[i, j]
                logfc[i, j] = de_test_temp.log_fold_change()
                logfc[j, i] = - logfc[i, j]
                if keep_full_test_objs:
                    tests[i, j] = de_test_temp
                    tests[j, i] = de_test_temp

        de_test = DifferentialExpressionTestPairwise(gene_ids=gene_names,
                                                     pval=pvals,
                                                     logfc=logfc,
                                                     ave=np.mean(X, axis=0),
                                                     groups=groups,
                                                     tests=tests,
                                                     correction_type=pval_correction)

    return de_test


def versus_rest(
        data,
        grouping: Union[str, np.ndarray, list],
        test: str = 'wald',
        gene_names: str = None,
        sample_description: pd.DataFrame = None,
        noise_model: str = None,
        pval_correction: str = "global",
        batch_size: int = None,
        training_strategy: Union[str, List[Dict[str, object]], Callable] = "AUTO",
        quick_scale: bool = None,
        keep_full_test_objs: bool = False,
        **kwargs
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
        
    :param data: input data
    :param grouping: str, array
    
        - column in data.obs/sample_description which contains the split of observations into the two groups.
        - array of length `num_observations` containing group labels
    :param test: str, statistical test to use. Possible options:
    
        - 'wald'
        - 'lrt'
        - 't-test'
        - 'wilcoxon'
    :param gene_names: optional list/array of gene names which will be used if `data` does not implicitly store these
    :param sample_description: optional pandas.DataFrame containing sample annotations
    :param noise_model: str, noise model to use in model-based unit_test. Possible options:
        
        - 'nb': default
    :param pval_correction: Choose between global and test-wise correction.
        Can be:

        - "global": correct all p-values in one operation
        - "by_test": correct the p-values of each test individually
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
    :param quick_scale: Depending on the optimizer, `scale` will be fitted faster and maybe less accurate.

        Useful in scenarios where fitting the exact `scale` is not absolutely necessary.
    :param keep_full_test_objs: [Debugging] keep the individual test objects; currently valid for test != "z-test"
    :param kwargs: [Debugging] Additional arguments will be passed to the _fit method.
    """
    if len(kwargs) != 0:
        logger.info("additional kwargs: %s", str(kwargs))

    # Do not store all models but only p-value and q-value matrix:
    # genes x groups
    X = _parse_data(data, gene_names)
    gene_names = _parse_gene_names(data, gene_names)
    sample_description = _parse_sample_description(data, sample_description)
    grouping = _parse_grouping(data, sample_description, grouping)
    sample_description = pd.DataFrame({"grouping": grouping})

    groups = np.unique(grouping)
    pvals = np.zeros([1, len(groups), X.shape[1]])
    logfc = np.zeros([1, len(groups), X.shape[1]])

    if keep_full_test_objs:
        tests = np.tile([None], [1, len(groups), X.shape[1]])
    else:
        tests = None

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
            quick_scale=quick_scale,
            **kwargs
        )
        pvals[0, i] = de_test_temp.pval
        logfc[0, i] = de_test_temp.log_fold_change()
        if keep_full_test_objs:
            tests[0, i] = de_test_temp

    de_test = DifferentialExpressionTestVsRest(gene_ids=gene_names,
                                               pval=pvals,
                                               logfc=logfc,
                                               ave=np.mean(X, axis=0),
                                               groups=groups,
                                               tests=tests,
                                               correction_type=pval_correction)

    return de_test

def partition(
    data,
    partition: Union[str, np.ndarray, list],
    gene_names: str = None,
    sample_description: pd.DataFrame = None):
    """
    Perform differential expression test for each group. This class handles 
    the partitioning of the data set, the differential test callls and
    the sumamry of the individual tests into one 
    DifferentialExpressionTestMulti object. All functions the yield
    DifferentialExpressionTestSingle objects can be performed on each 
    partition.

    Wraps _Partition so that doc strings are nice.

    :param data: input data
    :param grouping: str, array
    
        - column in data.obs/sample_description which contains the split of observations into the two groups.
        - array of length `num_observations` containing group labels
    :param gene_names: optional list/array of gene names which will be used if `data` does not implicitly store these
    :param sample_description: optional pandas.DataFrame containing sample annotations
    """
    return(_Partition(
        data=data,
        partition=partition,
        gene_names=gene_names,
        sample_description=sample_description))

class _Partition():
    """
    Perform differential expression test for each group. This class handles 
    the partitioning of the data set, the differential test callls and
    the sumamry of the individual tests into one 
    DifferentialExpressionTestMulti object. All functions the yield
    DifferentialExpressionTestSingle objects can be performed on each 
    partition.
    """

    def __init__(
        self,
        data,
        partition: Union[str, np.ndarray, list],
        gene_names: str = None,
        sample_description: pd.DataFrame = None):
        """
        :param data: input data
        :param partition: str, array
        
            - column in data.obs/sample_description which contains the split of observations into the two groups.
            - array of length `num_observations` containing group labels
        :param gene_names: optional list/array of gene names which will be used if `data` does not implicitly store these
        :param sample_description: optional pandas.DataFrame containing sample annotations
        """
        self.X = _parse_data(data, gene_names)
        self.gene_names = _parse_gene_names(data, gene_names)
        self.sample_description = _parse_sample_description(data, sample_description)
        self.partition = _parse_grouping(data, sample_description, partition)
        self.partitions = np.unique(self.partition)
        self.partition_idx = [np.where(self.partition==x)[0] for x in self.partitions]    

    def two_sample(
        self,
        grouping: Union[str],
        test=None,
        noise_model: str = None,
        batch_size: int = None,
        training_strategy: Union[str, List[Dict[str, object]], Callable] = "AUTO",
        **kwargs
    ) -> _DifferentialExpressionTestSingle:
        """
        See annotation of de.test.two_sample()
            
        :param grouping: str
        
            - column in data.obs/sample_description which contains the split of observations into the two groups.
        :param test: str, statistical test to use. Possible options:
        
            - 'wald': default
            - 'lrt'
            - 't-test'
            - 'wilcoxon'
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
        :param kwargs: [Debugging] Additional arguments will be passed to the _fit method.
        """
        DETestsSingle = []
        for i,idx in enumerate(self.partition_idx):
            DETestsSingle.append(two_sample(
                data=self.X[idx,:],
                grouping=grouping,
                test=test,
                gene_names=self.gene_names,
                sample_description=self.sample_description.iloc[idx,:],
                noise_model = noise_model,
                batch_size = batch_size,
                training_strategy=training_strategy,
                **kwargs
            ))
        return DifferentialExpressionTestByPartition(
            partitions=self.partitions, 
            tests=DETestsSingle, 
            ave=np.mean(self.X, axis=0),
            correction_type="by_test")

    def t_test(
        self,
        grouping: Union[str]
    ):
        """
        See annotation of de.test.t_test()
        
        :param grouping: str
        
            - column in data.obs/sample_description which contains the split of observations into the two groups.
        """
        DETestsSingle = []
        for i,idx in enumerate(self.partition_idx):
            DETestsSingle.append(t_test(
                data=self.X[idx,:],
                grouping=grouping,
                gene_names=self.gene_names,
                sample_description=self.sample_description.iloc[idx,:]
            ))
        return DifferentialExpressionTestByPartition(
            partitions=self.partitions, 
            tests=DETestsSingle, 
            ave=np.mean(self.X, axis=0),
            correction_type="by_test")

    def wilcoxon(
        self,
        grouping: Union[str],
    ):
        """
        See annotation of de.test.wilcoxon()
        
        :param grouping: str, array
        
            - column in data.obs/sample_description which contains the split of observations into the two groups.
            - array of length `num_observations` containing group labels
        """
        DETestsSingle = []
        for i,idx in enumerate(self.partition_idx):
            DETestsSingle.append(wilcoxon(
                data=self.X[idx,:],
                grouping=grouping,
                gene_names=self.gene_names,
                sample_description=self.sample_description.iloc[idx,:]
            ))
        return DifferentialExpressionTestByPartition(
            partitions=self.partitions, 
            tests=DETestsSingle, 
            ave=np.mean(self.X, axis=0),
            correction_type="by_test")

    def lrt(
        self,
        reduced_formula: str = None,
        full_formula: str = None,
        reduced_formula_loc: str = None,
        full_formula_loc: str = None,
        reduced_formula_scale: str = None,
        full_formula_scale: str = None,
        noise_model="nb",
        batch_size: int = None,
        training_strategy: Union[str, List[Dict[str, object]], Callable] = "AUTO",
        **kwargs
    ):
        """
        See annotation of de.test.lrt()
            
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
        :param kwargs: [Debugging] Additional arguments will be passed to the _fit method.
        """
        DETestsSingle = []
        for i,idx in enumerate(self.partition_idx):
            DETestsSingle.append(lrt(
                data=self.X[idx,:],
                reduced_formula=reduced_formula,
                full_formula=full_formula,
                reduced_formula_loc=reduced_formula_loc,
                full_formula_loc=full_formula_loc,
                reduced_formula_scale=reduced_formula_scale,
                full_formula_scale=full_formula_scale,
                gene_names=self.gene_names,
                sample_description=self.sample_description.iloc[idx,:],
                noise_model=noise_model,
                batch_size=batch_size,
                training_strategy=training_strategy,
                **kwargs
            ))
        return DifferentialExpressionTestByPartition(
            partitions=self.partitions, 
            tests=DETestsSingle, 
            ave=np.mean(self.X, axis=0),
            correction_type="by_test")

    def wald(
        self,
        factor_loc_totest: str,
        coef_to_test: object = None,  # e.g. coef_to_test="B"
        formula: str = None,
        formula_loc: str = None,
        formula_scale: str = None,
        noise_model: str = "nb",
        batch_size: int = None,
        training_strategy: Union[str, List[Dict[str, object]], Callable] = "AUTO",
        **kwargs
    ):
        """
        This function performs a wald test within each partition of a data set.
        See annotation of de.test.wald()
            
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
        :param kwargs: [Debugging] Additional arguments will be passed to the _fit method.
        """
        DETestsSingle = []
        for i,idx in enumerate(self.partition_idx):
            DETestsSingle.append(wald(
                data=self.X[idx,:],
                factor_loc_totest=factor_loc_totest,
                coef_to_test=coef_to_test,  # e.g. coef_to_test="B"
                formula=formula,
                formula_loc=formula_loc,
                formula_scale=formula_scale,
                gene_names=self.gene_names,
                sample_description=self.sample_description.iloc[idx,:],
                noise_model=noise_model,
                batch_size=batch_size,
                training_strategy=training_strategy,
                **kwargs
            ))
        return DifferentialExpressionTestByPartition(
            partitions=self.partitions, 
            tests=DETestsSingle, 
            ave=np.mean(self.X, axis=0),
            correction_type="by_test")
