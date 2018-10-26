import abc
import logging
from typing import Union, Dict, Tuple, List, Set, Callable

import pandas as pd

import numpy as np
# import scipy.sparse

# import dask
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
        self._log_probs = None

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
    def log_probs(self):
        if self._log_probs is None:
            self._log_probs = self._ll().compute()
        return self._log_probs

    @property
    def mean(self):
        if self._mean is None:
            self._mean = self._ave().compute()
        return self._mean

    @property
    def pval(self):
        if self._pval is None:
            self._pval = self._test().copy()
        return self._pval

    @property
    def qval(self, method="fdr_bh"):
        if self._qval is None:
            self._qval = self._correction(method=method).copy()
        return self._qval

    def log10_pval_clean(self, log10_threshold=-30):
        """
        Return log10 transformed and cleaned p-values.

        NaN p-values are set to one and p-values below log10_threshold
        in log10 space are set to log10_threshold.

        :param log10_threshold: minimal log10 p-value to return.
        :return: Cleaned log10 transformed p-values.
        """
        pvals = np.reshape(self.pval, -1)
        pvals = np.nextafter(0, 1, out=pvals, where=pvals == 0)
        log10_pval_clean = np.log(pvals) / np.log(10)
        log10_pval_clean[np.isnan(log10_pval_clean)] = 1
        log10_pval_clean = np.clip(log10_pval_clean, log10_threshold, 0, log10_pval_clean)
        return log10_pval_clean

    def log10_qval_clean(self, log10_threshold=-30):
        """
        Return log10 transformed and cleaned q-values.

        NaN p-values are set to one and q-values below log10_threshold
        in log10 space are set to log10_threshold.

        :param log10_threshold: minimal log10 q-value to return.
        :return: Cleaned log10 transformed q-values.
        """
        qvals = np.reshape(self.qval, -1)
        qvals = np.nextafter(0, 1, out=qvals, where=qvals == 0)
        log10_qval_clean = np.log(qvals) / np.log(10)
        log10_qval_clean[np.isnan(log10_qval_clean)] = 1
        log10_qval_clean = np.clip(log10_qval_clean, log10_threshold, 0, log10_qval_clean)
        return log10_qval_clean

    @property
    @abc.abstractmethod
    def summary(self, **kwargs) -> pd.DataFrame:
        pass

    def _threshold_summary(self, res, qval_thres=None,
                           fc_upper_thres=None, fc_lower_thres=None, mean_thres=None) -> pd.DataFrame:
        """
        Reduce differential expression results into an output table with desired thresholds.
        """
        if qval_thres is not None:
            res = res.iloc[res['qval'].values <= qval_thres, :]

        if fc_upper_thres is not None and fc_lower_thres is None:
            res = res.iloc[res['log2fc'].values >= np.log(fc_upper_thres) / np.log(2), :]
        elif fc_upper_thres is None and fc_lower_thres is not None:
            res = res.iloc[res['log2fc'].values <= np.log(fc_lower_thres) / np.log(2), :]
        elif fc_upper_thres is not None and fc_lower_thres is not None:
            res = res.iloc[np.logical_or(
                res['log2fc'].values <= np.log(fc_lower_thres) / np.log(2),
                res['log2fc'].values >= np.log(fc_upper_thres) / np.log(2)), :]

        if mean_thres is not None:
            res = res.iloc[res['mean'].values >= mean_thres, :]

        return res

    def plot_volcano(self, log10_p_threshold=-30, log2_fc_threshold=10):
        """
        returns a volcano plot of p-value vs. log fold change

        :return: Tuple of matplotlib (figure, axis)
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        neg_log_pvals = - self.log10_pval_clean(log10_threshold=log10_p_threshold)
        logfc = np.reshape(self.log2_fold_change(), -1)
        logfc = np.clip(logfc, -log2_fc_threshold, log2_fc_threshold, logfc)

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

    def summary(self, qval_thres=None,
                fc_upper_thres=None, fc_lower_thres=None, mean_thres=None, **kwargs) -> pd.DataFrame:
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

    def summary(self, qval_thres=None, fc_upper_thres=None,
                fc_lower_thres=None, mean_thres=None,
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
            fc_upper_thres=fc_upper_thres,
            fc_lower_thres=fc_lower_thres,
            mean_thres=mean_thres
        )

        return res


class DifferentialExpressionTestWald(_DifferentialExpressionTestSingle):
    """
    Single wald test per gene.
    """

    model_estim: _Estimation
    sd_loc_totest: np.ndarray
    coef_loc_totest: np.ndarray
    indep_coefs: np.ndarray
    theta_mle: np.ndarray
    theta_sd: np.ndarray

    def __init__(
            self,
            model_estim: _Estimation,
            col_indices: np.ndarray,
            indep_coefs: np.ndarray = None
    ):
        """
        :param model_estim:
        :param cold_index: indices of indep_coefs to test
        :param indep_coefs: indices of independent coefficients in coefficient vector
        """
        super().__init__()
        self.model_estim = model_estim
        self.coef_loc_totest = col_indices
        # Note that self.indep_coefs are relevant if constraints are given
        # and hessian is computed across independent coefficients only 
        # whereas point estimators are given for all coefficients.
        if indep_coefs is not None:
            self.indep_coefs = indep_coefs
            self.sd_loc_totest = np.where(self.indep_coefs == col_indices)[0]
        else:
            self.sd_loc_totest = self.coef_loc_totest
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
    def gene_ids(self) -> np.ndarray:
        return np.asarray(self.model_estim.features)

    @property
    def model_gradient(self):
        return self.model_estim.gradient

    def log_fold_change(self, base=np.e, **kwargs):
        """
        Returns one fold change per gene

        Returns coefficient if only one coefficient is testeed.
        Returns mean absolute coefficient if multiple coefficients are tested.
        """
        # design = np.unique(self.model_estim.design_loc, axis=0)
        # dmat = np.zeros_like(design)
        # dmat[:, self.coef_loc_totest] = design[:, self.coef_loc_totest]

        # loc = dmat @ self.model_estim.par_link_loc[self.coef_loc_totest]
        # return loc[1] - loc[0]
        if len(self.coef_loc_totest) == 1:
            return self.model_estim.par_link_loc[self.coef_loc_totest][0]
        else:
            idx_max = np.argmax(np.abs(self.model_estim.par_link_loc[self.coef_loc_totest]), axis=0)
            return self.model_estim.par_link_loc[self.coef_loc_totest][
                idx_max, np.arange(self.model_estim.par_link_loc.shape[1])]

    def _ll(self):
        """
        Returns a xr.DataArray containing the log likelihood of each gene

        :return: xr.DataArray
        """
        return np.sum(self.model_estim.log_probs(), axis=0)

    def _ave(self):
        """
        Returns a xr.DataArray containing the mean expression by gene

        :return: xr.DataArray
        """
        return np.mean(self.model_estim.X, axis=0)

    def _test(self):
        """
        Returns a xr.DataArray containing the p-value for differential expression for each gene

        :return: xr.DataArray
        """
        # Check whether single- or multiple parameters are tested.
        # For a single parameter, the wald statistic distribution is approximated
        # with a normal distribution, for multiple parameters, a chi-square distribution is used.
        self.theta_mle = self.model_estim.par_link_loc[self.coef_loc_totest]
        if len(self.coef_loc_totest) == 1:
            self.theta_mle = self.theta_mle[0]  # Make xarray one dimensinoal for stats.wald_test.
            self.theta_sd = self.model_estim.fisher_inv[:, self.sd_loc_totest[0], self.sd_loc_totest[0]].values
            self.theta_sd = np.nextafter(0, np.inf, out=self.theta_sd,
                                         where=self.theta_sd < np.nextafter(0, np.inf))
            self.theta_sd = np.sqrt(self.theta_sd)
            return stats.wald_test(
                theta_mle=self.theta_mle,
                theta_sd=self.theta_sd,
                theta0=0
            )
        else:
            # We avoid inverting the covariance matrix (FIM) here by directly feeding
            # its inverse, the negative hessian, to wald_test_chisq. Note that 
            # the negative hessian is pre-computed within batchglm.
            self.theta_sd = np.diagonal(self.model_estim.fisher_inv, axis1=-2, axis2=-1).copy()
            self.theta_sd = np.nextafter(0, np.inf, out=self.theta_sd,
                                         where=self.theta_sd < np.nextafter(0, np.inf))
            self.theta_sd = np.sqrt(self.theta_sd)
            return stats.wald_test_chisq(
                theta_mle=self.theta_mle,
                theta_invcovar=-self.model_estim.hessians[:, self.sd_loc_totest, self.sd_loc_totest],
                theta0=0
            )

    def summary(self, qval_thres=None, fc_upper_thres=None,
                fc_lower_thres=None, mean_thres=None,
                **kwargs) -> pd.DataFrame:
        """
        Summarize differential expression results into an output table.
        """
        res = super().summary(**kwargs)
        res["grad"] = self.model_gradient.data
        if len(self.theta_mle.shape) == 1:
            res["coef_mle"] = self.theta_mle
        if len(self.theta_sd.shape) == 1:
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
            fc_upper_thres=fc_upper_thres,
            fc_lower_thres=fc_lower_thres,
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
            gene_names=self.gene_ids,
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

        mean_x1 = np.mean(x1, axis=0)
        mean_x1 = mean_x1.clip(np.nextafter(0, 1), np.inf)
        mean_x2 = np.mean(x1, axis=0)
        mean_x2 = mean_x2.clip(np.nextafter(0, 1), np.inf)

        self._logfc = np.log(mean_x1) - np.log(mean_x2).data
        # Return 0 if LFC was non-zero and variances are zero,
        # this causes division by zero in the test statistic. This
        # is a highly significant result if one believes the variance estimate.
        pval[np.logical_and(np.logical_and(self._var_geq_zero == False,
                                           self._ave_geq_zero == True),
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

    def summary(self, qval_thres=None, fc_upper_thres=None,
                fc_lower_thres=None, mean_thres=None,
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
            fc_upper_thres=fc_upper_thres,
            fc_lower_thres=fc_lower_thres,
            mean_thres=mean_thres
        )

        return res


class DifferentialExpressionTestWilcoxon(_DifferentialExpressionTestSingle):
    """
    Single wilcoxon rank sum test per gene.
    """

    def __init__(self, data, grouping, gene_names):
        super().__init__()
        self.data = data
        self.grouping = grouping
        self._gene_names = np.asarray(gene_names)

        x0, x1 = _split_X(data, grouping)

        self._mean = np.mean(data, axis=0)
        self._pval = stats.wilcoxon_test(x0=x0.data, x1=x1.data)
        self._logfc = np.log(np.mean(x1, axis=0)) - np.log(np.mean(x0, axis=0)).data
        q = self.qval

    @property
    def gene_ids(self) -> np.ndarray:
        return self._gene_names

    def log_fold_change(self, base=np.e, **kwargs):
        """
        Returns one fold change per gene
        """
        if base == np.e:
            return self._logfc
        else:
            return self._logfc / np.log(base)

    def summary(self, qval_thres=None, fc_upper_thres=None,
                fc_lower_thres=None, mean_thres=None,
                **kwargs) -> pd.DataFrame:
        """
        Summarize differential expression results into an output table.
        """
        res = super().summary(**kwargs)

        res = self._threshold_summary(
            res=res,
            qval_thres=qval_thres,
            fc_upper_thres=fc_upper_thres,
            fc_lower_thres=fc_lower_thres,
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
        """
        Get p-values of the comparison of group1 and group2.

        :param group1: Identifier of first group of observations in pair-wise comparison.
        :param group2: Identifier of second group of observations in pair-wise comparison.
        :return: p-values
        """
        assert self._pval is not None

        self._check_groups(group1, group2)
        return self._pval[self.groups.index(group1), self.groups.index(group2), :]

    def qval_pair(self, group1, group2):
        """
        Get q-values of the comparison of group1 and group2.

        :param group1: Identifier of first group of observations in pair-wise comparison.
        :param group2: Identifier of second group of observations in pair-wise comparison.
        :return: q-values
        """
        assert self._qval is not None

        self._check_groups(group1, group2)
        return self._qval[self.groups.index(group1), self.groups.index(group2), :]

    def log10_pval_pair_clean(self, group1, group2, log10_threshold=-30):
        """
        Return log10 transformed and cleaned p-values.

        NaN p-values are set to one and p-values below log10_threshold
        in log10 space are set to log10_threshold.

        :param group1: Identifier of first group of observations in pair-wise comparison.
        :param group2: Identifier of second group of observations in pair-wise comparison.
        :param log10_threshold: minimal log10 p-value to return.
        :return: Cleaned log10 transformed p-values.
        """
        pvals = np.reshape(self.pval_pair(group1=group1, group2=group2), -1)
        pvals = np.nextafter(0, 1, out=pvals, where=pvals == 0)
        log10_pval_clean = np.log(pvals) / np.log(10)
        log10_pval_clean[np.isnan(log10_pval_clean)] = 1
        log10_pval_clean = np.clip(log10_pval_clean, log10_threshold, 0, log10_pval_clean)
        return log10_pval_clean

    def log10_qval_pair_clean(self, group1, group2, log10_threshold=-30):
        """
        Return log10 transformed and cleaned q-values.

        NaN p-values are set to one and q-values below log10_threshold
        in log10 space are set to log10_threshold.

        :param group1: Identifier of first group of observations in pair-wise comparison.
        :param group2: Identifier of second group of observations in pair-wise comparison.
        :param log10_threshold: minimal log10 q-value to return.
        :return: Cleaned log10 transformed q-values.
        """
        qvals = np.reshape(self.qval_pair(group1=group1, group2=group2), -1)
        qvals = np.nextafter(0, 1, out=qvals, where=qvals == 0)
        log10_qval_clean = np.log(qvals) / np.log(10)
        log10_qval_clean[np.isnan(log10_qval_clean)] = 1
        log10_qval_clean = np.clip(log10_qval_clean, log10_threshold, 0, log10_qval_clean)
        return log10_qval_clean

    def log_fold_change_pair(self, group1, group2, base=np.e):
        """
        Get log fold changes of the comparison of group1 and group2.

        :param group1: Identifier of first group of observations in pair-wise comparison.
        :param group2: Identifier of second group of observations in pair-wise comparison.
        :return: log fold changes
        """
        assert self._logfc is not None

        self._check_groups(group1, group2)
        return self.log_fold_change(base=base)[self.groups.index(group1), self.groups.index(group2), :]

    def summary(self, qval_thres=None, fc_upper_thres=None,
                fc_lower_thres=None, mean_thres=None,
                **kwargs) -> pd.DataFrame:
        """
        Summarize differential expression results into an output table.
        """
        res = super().summary(**kwargs)

        res = self._threshold_summary(
            res=res,
            qval_thres=qval_thres,
            fc_upper_thres=fc_upper_thres,
            fc_lower_thres=fc_lower_thres,
            mean_thres=mean_thres
        )

        return res

    def summary_pair(self, group1, group2,
                     qval_thres=None, fc_upper_thres=None,
                     fc_lower_thres=None, mean_thres=None,
                     **kwargs) -> pd.DataFrame:
        """
        Summarize differential expression results into an output table.

        :param group1: Identifier of first group of observations in pair-wise comparison.
        :param group2: Identifier of second group of observations in pair-wise comparison.
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

        res = self._threshold_summary(
            res=res,
            qval_thres=qval_thres,
            fc_upper_thres=fc_upper_thres,
            fc_lower_thres=fc_lower_thres,
            mean_thres=mean_thres
        )

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
        return self.pval[self.groups.index(group1), self.groups.index(group2), :]

    def qval_pair(self, group1, group2):
        self._check_groups(group1, group2)
        return self.qval[self.groups.index(group1), self.groups.index(group2), :]

    def log_fold_change_pair(self, group1, group2, base=np.e):
        self._check_groups(group1, group2)
        return self.log_fold_change(base=base)[self.groups.index(group1), self.groups.index(group2), :]

    def summary(self, qval_thres=None, fc_upper_thres=None,
                fc_lower_thres=None, mean_thres=None,
                **kwargs) -> pd.DataFrame:
        """
        Summarize differential expression results into an output table.
        """
        res = super().summary(**kwargs)

        res = self._threshold_summary(
            res=res,
            qval_thres=qval_thres,
            fc_upper_thres=fc_upper_thres,
            fc_lower_thres=fc_lower_thres,
            mean_thres=mean_thres
        )

        return res

    def summary_pair(self, group1, group2,
                     qval_thres=None, fc_upper_thres=None,
                     fc_lower_thres=None, mean_thres=None,
                     **kwargs) -> pd.DataFrame:
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

        res = self._threshold_summary(
            res=res,
            qval_thres=qval_thres,
            fc_upper_thres=fc_upper_thres,
            fc_lower_thres=fc_lower_thres,
            mean_thres=mean_thres
        )

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
            raise ValueError('group "%s" not recognized' % group)

    def pval_group(self, group):
        self._check_group(group)
        return self.pval[0, self.groups.index(group), :]

    def qval_group(self, group):
        self._check_group(group)
        return self.qval[0, self.groups.index(group), :]

    def log_fold_change_group(self, group, base=np.e):
        self._check_group(group)
        return self.log_fold_change(base=base)[0, self.groups.index(group), :]

    def summary(self, qval_thres=None, fc_upper_thres=None,
                fc_lower_thres=None, mean_thres=None,
                **kwargs) -> pd.DataFrame:
        """
        Summarize differential expression results into an output table.
        """
        res = super().summary(**kwargs)

        res = self._threshold_summary(
            res=res,
            qval_thres=qval_thres,
            fc_upper_thres=fc_upper_thres,
            fc_lower_thres=fc_lower_thres,
            mean_thres=mean_thres
        )

        return res

    def summary_group(self, group,
                      qval_thres=None, fc_upper_thres=None,
                      fc_lower_thres=None, mean_thres=None,
                      **kwargs) -> pd.DataFrame:
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

        res = self._threshold_summary(
            res=res,
            qval_thres=qval_thres,
            fc_upper_thres=fc_upper_thres,
            fc_lower_thres=fc_lower_thres,
            mean_thres=mean_thres
        )

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
            raise ValueError('partition "%s" not recognized' % partition)

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

    def summary(self, qval_thres=None, fc_upper_thres=None,
                fc_lower_thres=None, mean_thres=None,
                **kwargs) -> pd.DataFrame:
        """
        Summarize differential expression results into an output table.
        """
        res = super().summary(**kwargs)

        res = self._threshold_summary(
            res=res,
            qval_thres=qval_thres,
            fc_upper_thres=fc_upper_thres,
            fc_lower_thres=fc_lower_thres,
            mean_thres=mean_thres
        )

        return res


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


def _parse_data(data, gene_names) -> xr.DataArray:
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
    assert data.shape[0] == sample_description.shape[
        0], "data matrix and sample description must contain same number of cells"
    return sample_description


def _parse_size_factors(size_factors, data):
    if size_factors is not None:
        if isinstance(size_factors, pd.core.series.Series):
            size_factors = size_factors.values
        assert size_factors.shape[0] == data.shape[0], "data matrix and size factors must contain same number of cells"
    return size_factors


def design_matrix(
        data=None,
        sample_description: pd.DataFrame = None,
        formula: str = None,
        dmat: pd.DataFrame = None
) -> Union[patsy.design_info.DesignMatrix, xr.Dataset]:
    """ Build design matrix for fit of generalized linear model.

    This is necessary for wald tests and likelihood ratio tests.
    This function only carries through formatting if dmat is directly supplied.

    :param data: input data
    :param formula: model formula.
    :param sample_description: optional pandas.DataFrame containing sample annotations
    :param dmat: model design matrix
    """
    if data is None and sample_description is None and dmat is None:
        raise ValueError("Supply either data or sample_description or dmat.")
    if dmat is None and formula is None:
        raise ValueError("Supply either dmat or formula.")

    if dmat is None:
        sample_description = _parse_sample_description(data, sample_description)
        dmat = data_utils.design_matrix(sample_description=sample_description, formula=formula)

        return dmat
    else:
        ar = xr.DataArray(dmat, dims=("observations", "design_params"))
        ar.coords["design_params"] = dmat.columns

        ds = xr.Dataset({
            "design": ar,
        })

        return ds


def coef_names(
        data=None,
        sample_description: pd.DataFrame = None,
        formula: str = None,
        dmat: pd.DataFrame = None
) -> list:
    """ Output coefficient names of model only.

    :param data: input data
    :param formula: model formula.
    :param sample_description: optional pandas.DataFrame containing sample annotations
    :param dmat: model design matrix
    """
    return design_matrix(
        data=data,
        sample_description=sample_description,
        formula=formula,
        dmat=dmat
    ).design_info.column_names


def _fit(
        noise_model,
        data,
        design_loc,
        design_scale,
        constraints_loc: np.ndarray = None,
        constraints_scale: np.ndarray = None,
        init_model=None,
        gene_names=None,
        size_factors=None,
        batch_size: int = None,
        training_strategy: Union[str, List[Dict[str, object]], Callable] = "AUTO",
        quick_scale: bool = None,
        close_session=True,
        dtype="float32"
):
    """
    :param noise_model: str, noise model to use in model-based unit_test. Possible options:

        - 'nb': default
    :param design_loc: Design matrix of location model.
    :param design_loc: Design matrix of scale model.
    :param constraints_loc: : Constraints for location model.
        Array with constraints in rows and model parameters in columns.
        Each constraint contains non-zero entries for the a of parameters that 
        has to sum to zero. This constraint is enforced by binding one parameter
        to the negative sum of the other parameters, effectively representing that
        parameter as a function of the other parameters. This dependent
        parameter is indicated by a -1 in this array, the independent parameters
        of that constraint (which may be dependent at an earlier constraint)
        are indicated by a 1.
    :param constraints_scale: : Constraints for scale model.
        Array with constraints in rows and model parameters in columns.
        Each constraint contains non-zero entries for the a of parameters that 
        has to sum to zero. This constraint is enforced by binding one parameter
        to the negative sum of the other parameters, effectively representing that
        parameter as a function of the other parameters. This dependent
        parameter is indicated by a -1 in this array, the independent parameters
        of that constraint (which may be dependent at an earlier constraint)
        are indicated by a 1.
    :param size_factors: 1D array of transformed library size factors for each cell in the 
        same order as in data
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
    :param dtype: Allows specifying the precision which should be used to fit data.

        Should be "float32" for single precision or "float64" for double precision.
    :param close_session: If True, will finalize the estimator. Otherwise, return the estimator itself.
    """
    if isinstance(training_strategy, str) and training_strategy.lower() == 'bfgs':
        lib_size = np.zeros(data.shape[0])
        if noise_model == "nb" or noise_model == "negative_binomial":
            estim = Estim_BFGS(X=data, design_loc=design_loc, design_scale=design_scale,
                               lib_size=lib_size, batch_size=batch_size, feature_names=gene_names)
            estim.run(nproc=3, maxiter=10000, debug=False)
            model = estim.return_batchglm_formated_model()
        else:
            raise ValueError('base.test(): `noise_model="%s"` not recognized.' % noise_model)
    else:
        if noise_model == "nb" or noise_model == "negative_binomial":
            import batchglm.api.models.nb_glm as test_model

            logger.info("Fitting model...")
            logger.debug(" * Assembling input data...")
            input_data = test_model.InputData.new(
                data=data,
                design_loc=design_loc,
                design_scale=design_scale,
                constraints_loc=constraints_loc,
                constraints_scale=constraints_scale,
                size_factors=size_factors,
                feature_names=gene_names,
            )

            logger.debug(" * Set up Estimator...")
            constructor_args = {}
            if batch_size is not None:
                constructor_args["batch_size"] = batch_size
            if quick_scale is not None:
                constructor_args["quick_scale"] = quick_scale
            estim = test_model.Estimator(
                input_data=input_data,
                init_model=init_model,
                dtype=dtype,
                **constructor_args
            )

            logger.debug(" * Initializing Estimator...")
            estim.initialize()

            logger.debug(" * Run estimation...")
            # training:
            if callable(training_strategy):
                # call training_strategy if it is a function
                training_strategy(estim)
            else:
                estim.train_sequence(training_strategy)

            if close_session:
                logger.debug(" * Finalize estimation...")
                model = estim.finalize()
            else:
                model = estim
            logger.debug(" * Model fitting done.")

        else:
            raise ValueError('base.test(): `noise_model="%s"` not recognized.' % noise_model)

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
        size_factors: np.ndarray = None,
        batch_size: int = None,
        training_strategy: Union[str, List[Dict[str, object]], Callable] = "AUTO",
        quick_scale: bool = None,
        dtype="float64",
        **kwargs
):
    """
    Perform log-likelihood ratio test for differential expression for each gene.

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
    :param size_factors: 1D array of transformed library size factors for each cell in the 
        same order as in data
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
    :param dtype: Allows specifying the precision which should be used to fit data.

        Should be "float32" for single precision or "float64" for double precision.
    :param kwargs: [Debugging] Additional arguments will be passed to the _fit method.
    """
    if len(kwargs) != 0:
        logger.info("additional kwargs: %s", str(kwargs))

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
    size_factors = _parse_size_factors(size_factors=size_factors, data=X)

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
        size_factors=size_factors,
        batch_size=batch_size,
        training_strategy=training_strategy,
        quick_scale=quick_scale,
        dtype=dtype,
        **kwargs
    )
    full_model = _fit(
        noise_model=noise_model,
        data=X,
        design_loc=full_design_loc,
        design_scale=full_design_scale,
        gene_names=gene_names,
        init_model=reduced_model,
        size_factors=size_factors,
        batch_size=batch_size,
        # batch_size=X.shape[0],  # workaround: batch_size=num_observations
        training_strategy=training_strategy,
        quick_scale=quick_scale,
        dtype=dtype,
        **kwargs
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
        factor_loc_totest: Union[str, List[str]] = None,
        coef_to_test: Union[str, List[str]] = None,  # e.g. coef_to_test="B"
        formula: str = None,
        formula_loc: str = None,
        formula_scale: str = None,
        gene_names: Union[str, np.ndarray] = None,
        sample_description: pd.DataFrame = None,
        dmat_loc: Union[patsy.design_info.DesignMatrix, xr.Dataset] = None,
        dmat_scale: Union[patsy.design_info.DesignMatrix, xr.Dataset] = None,
        constraints_loc: np.ndarray = None,
        constraints_scale: np.ndarray = None,
        noise_model: str = "nb",
        size_factors: np.ndarray = None,
        batch_size: int = None,
        training_strategy: Union[str, List[Dict[str, object]], Callable] = "AUTO",
        quick_scale: bool = None,
        dtype="float64",
        **kwargs
):
    """
    Perform Wald test for differential expression for each gene.

    :param data: input data
    :param factor_loc_totest:
        List of factors of formula to test with Wald test.
        E.g. "condition" or ["batch", "condition"] if formula_loc would be "~ 1 + batch + condition"
    :param coef_to_test:
        If there are more than two groups specified by `factor_loc_totest`,
        this parameter allows to specify the group which should be tested.
        Alternatively, if factor_loc_totest is not given, this list sets
        the exact coefficients which are to be tested.
    :param formula: formula
        model formula for location and scale parameter models.
    :param formula_loc: formula
        model formula for location and scale parameter models.
        If not specified, `formula` will be used instead.
    :param formula_scale: formula
        model formula for scale parameter model.
        If not specified, `formula` will be used instead.
    :param gene_names: optional list/array of gene names which will be used if `data` does not implicitly store these
    :param sample_description: optional pandas.DataFrame containing sample annotations
    :param dmat_loc: Pre-built location model design matrix. 
        This over-rides formula_loc and sample description information given in
        data or sample_description. 
    :param dmat_scale: Pre-built scale model design matrix.
        This over-rides formula_scale and sample description information given in
        data or sample_description.
    :param constraints_loc: : Constraints for location model.
        Array with constraints in rows and model parameters in columns.
        Each constraint contains non-zero entries for the a of parameters that 
        has to sum to zero. This constraint is enforced by binding one parameter
        to the negative sum of the other parameters, effectively representing that
        parameter as a function of the other parameters. This dependent
        parameter is indicated by a -1 in this array, the independent parameters
        of that constraint (which may be dependent at an earlier constraint)
        are indicated by a 1. It is highly recommended to only use this option
        together with prebuilt design matrix for the location model, dmat_loc.
    :param constraints_scale: : Constraints for scale model.
        Array with constraints in rows and model parameters in columns.
        Each constraint contains non-zero entries for the a of parameters that 
        has to sum to zero. This constraint is enforced by binding one parameter
        to the negative sum of the other parameters, effectively representing that
        parameter as a function of the other parameters. This dependent
        parameter is indicated by a -1 in this array, the independent parameters
        of that constraint (which may be dependent at an earlier constraint)
        are indicated by a 1. It is highly recommended to only use this option
        together with prebuilt design matrix for the scale model, dmat_scale.
    :param size_factors: 1D array of transformed library size factors for each cell in the 
        same order as in data
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
    :param dtype: Allows specifying the precision which should be used to fit data.

        Should be "float32" for single precision or "float64" for double precision.
    :param kwargs: [Debugging] Additional arguments will be passed to the _fit method.
    """
    if len(kwargs) != 0:
        logger.debug("additional kwargs: %s", str(kwargs))

    if formula_loc is None:
        formula_loc = formula
    if formula_scale is None:
        formula_scale = formula
    if dmat_loc is None and formula_loc is None:
        raise ValueError("Supply either dmat_loc or formula_loc or formula.")
    if dmat_scale is None and formula_scale is None:
        raise ValueError("Supply either dmat_loc or formula_loc or formula.")
    # Check that factor_loc_totest and coef_to_test are lists and not single strings:
    if isinstance(factor_loc_totest, str):
        factor_loc_totest = [factor_loc_totest]
    if isinstance(coef_to_test, str):
        coef_to_test = [coef_to_test]

    # # Parse input data formats:
    X = _parse_data(data, gene_names)
    gene_names = _parse_gene_names(data, gene_names)
    if dmat_loc is None and dmat_scale is None:
        sample_description = _parse_sample_description(data, sample_description)
    size_factors = _parse_size_factors(size_factors=size_factors, data=X)

    if dmat_loc is None:
        design_loc = data_utils.design_matrix(
            sample_description=sample_description, formula=formula_loc)
    else:
        design_loc = dmat_loc

    if dmat_scale is None:
        design_scale = data_utils.design_matrix(
            sample_description=sample_description, formula=formula_scale)
    else:
        design_scale = dmat_scale

    # Coefficients to test:
    indep_coef_indices = None
    col_indices = None
    if factor_loc_totest is not None:
        # Select coefficients to test via formula model:
        col_indices = np.concatenate([
            np.arange(design_loc.shape[-1])[design_loc.design_info.slice(x)]
            for x in factor_loc_totest
        ])
        assert col_indices.size > 0, "Could not find any matching columns!"
        if coef_to_test is not None:
            if len(factor_loc_totest) > 1:
                raise ValueError("do not set coef_to_test if more than one factor_loc_totest is given")
            samples = sample_description[factor_loc_totest].astype(type(coef_to_test)) == coef_to_test
            one_cols = np.where(design_loc[samples][:, col_indices][0] == 1)
            if one_cols.size == 0:
                # there is no such column; modify design matrix to create one
                design_loc[:, col_indices] = np.where(samples, 1, 0)
    elif coef_to_test is not None:
        # Directly select coefficients to test from design matrix (xarray):
        # Check that coefficients to test are not dependent parameters if constraints are given:
        # TODO: design_loc is sometimes xarray and sometimes patsy when it arrives here, 
        # should it not always be xarray?
        if isinstance(design_loc, patsy.design_info.DesignMatrix):
            col_indices = np.asarray([
                design_loc.design_info.column_names.index(x)
                for x in coef_to_test
            ])
        else:
            col_indices = np.asarray([
                list(np.asarray(design_loc.coords['design_params'])).index(x)
                for x in coef_to_test
            ])
        if constraints_loc is not None:
            dep_coef_indices = np.where(np.any(constraints_loc == -1, axis=0) == True)[0]
            assert np.all([x not in dep_coef_indices for x in col_indices]), "cannot test dependent coefficient"
            indep_coef_indices = np.where(np.any(constraints_loc == -1, axis=0) == False)[0]

    ## Fit GLM:
    model = _fit(
        noise_model=noise_model,
        data=X,
        design_loc=design_loc,
        design_scale=design_scale,
        constraints_loc=constraints_loc,
        constraints_scale=constraints_scale,
        gene_names=gene_names,
        size_factors=size_factors,
        batch_size=batch_size,
        training_strategy=training_strategy,
        quick_scale=quick_scale,
        dtype=dtype,
        **kwargs,
    )

    ## Perform DE test:
    de_test = DifferentialExpressionTestWald(
        model,
        col_indices=col_indices,
        indep_coefs=indep_coef_indices
    )

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
        sample_description=None,
        dtype="float32"
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
    X: xr.DataArray = _parse_data(data, gene_names)
    grouping = _parse_grouping(data, sample_description, grouping)

    de_test = DifferentialExpressionTestTT(
        data=X.astype(dtype),
        grouping=grouping,
        gene_ids=gene_names,
    )

    return de_test


def wilcoxon(
        data,
        grouping,
        gene_names=None,
        sample_description=None,
        dtype="float32"
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
    X: xr.DataArray = _parse_data(data, gene_names)
    grouping = _parse_grouping(data, sample_description, grouping)

    de_test = DifferentialExpressionTestWilcoxon(
        data=X.astype(dtype),
        grouping=grouping,
        gene_names=gene_names,
    )

    return de_test


def two_sample(
        data,
        grouping: Union[str, np.ndarray, list],
        test=None,
        gene_names=None,
        sample_description=None,
        noise_model: str = None,
        size_factors: np.ndarray = None,
        batch_size: int = None,
        training_strategy: Union[str, List[Dict[str, object]], Callable] = "AUTO",
        quick_scale: bool = None,
        dtype="float32",
        **kwargs
) -> _DifferentialExpressionTestSingle:
    r"""
    Perform differential expression test between two groups on adata object
    for each gene.

    This function wraps the selected statistical test for the scenario of
    a two sample comparison. All unit_test offered in this wrapper
    test for the difference of the mean parameter of both samples.
    The exact unit_test are as follows (assuming the group labels
    are saved in a column named "group"):

    - lrt(log-likelihood ratio test):
        Requires the fitting of 2 generalized linear models (full and reduced).
        The models are automatically assembled as follows, use the de.test.lrt()
        function if you would like to perform a different test.

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
    :param size_factors: 1D array of transformed library size factors for each cell in the 
        same order as in data
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
    :param dtype: Allows specifying the precision which should be used to fit data.

        Should be "float32" for single precision or "float64" for double precision.
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
            size_factors=size_factors,
            batch_size=batch_size,
            training_strategy=training_strategy,
            quick_scale=quick_scale,
            dtype=dtype,
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
            size_factors=size_factors,
            batch_size=batch_size,
            training_strategy=training_strategy,
            quick_scale=quick_scale,
            dtype=dtype,
            **kwargs
        )
    elif test.lower() == 't-test' or test.lower() == "t_test" or test.lower() == "ttest":
        de_test = t_test(
            data=X,
            gene_names=gene_names,
            grouping=grouping,
            dtype=dtype
        )
    elif test.lower() == 'wilcoxon':
        de_test = wilcoxon(
            data=X,
            gene_names=gene_names,
            grouping=grouping,
            dtype=dtype
        )
    else:
        raise ValueError('base.two_sample(): Parameter `test="%s"` not recognized.' % test)

    return de_test


def pairwise(
        data,
        grouping: Union[str, np.ndarray, list],
        test: str = 'z-test',
        gene_names: str = None,
        sample_description: pd.DataFrame = None,
        noise_model: str = None,
        pval_correction: str = "global",
        size_factors: np.ndarray = None,
        batch_size: int = None,
        training_strategy: Union[str, List[Dict[str, object]], Callable] = "AUTO",
        quick_scale: bool = None,
        dtype="float32",
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
    :param size_factors: 1D array of transformed library size factors for each cell in the 
        same order as in data
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
    :param dtype: Allows specifying the precision which should be used to fit data.

        Should be "float32" for single precision or "float64" for double precision.
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
            size_factors=size_factors,
            batch_size=batch_size,
            training_strategy=training_strategy,
            quick_scale=quick_scale,
            dtype=dtype,
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
            tests = np.tile([None], [len(groups), len(groups)])
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
                    size_factors=size_factors[sel] if size_factors is not None else None,
                    batch_size=batch_size,
                    training_strategy=training_strategy,
                    quick_scale=quick_scale,
                    dtype=dtype,
                    **kwargs
                )
                pvals[i, j] = de_test_temp.pval
                pvals[j, i] = pvals[i, j]
                logfc[i, j] = de_test_temp.log_fold_change()
                logfc[j, i] = - logfc[i, j]
                if keep_full_test_objs:
                    tests[i, j] = de_test_temp
                    tests[j, i] = de_test_temp

        de_test = DifferentialExpressionTestPairwise(
            gene_ids=gene_names,
            pval=pvals,
            logfc=logfc,
            ave=np.mean(X, axis=0),
            groups=groups,
            tests=tests,
            correction_type=pval_correction
        )

    return de_test


def versus_rest(
        data,
        grouping: Union[str, np.ndarray, list],
        test: str = 'wald',
        gene_names: str = None,
        sample_description: pd.DataFrame = None,
        noise_model: str = None,
        pval_correction: str = "global",
        size_factors: np.ndarray = None,
        batch_size: int = None,
        training_strategy: Union[str, List[Dict[str, object]], Callable] = "AUTO",
        quick_scale: bool = None,
        dtype="float32",
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
    :param size_factors: 1D array of transformed library size factors for each cell in the 
        same order as in data
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

        Useful in scenarios where fitting the exact `scale` is not
    :param dtype: Allows specifying the precision which should be used to fit data.

        Should be "float32" for single precision or "float64" for double precision.
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
        tests = np.tile([None], [1, len(groups)])
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
            size_factors=size_factors,
            dtype=dtype,
            **kwargs
        )
        pvals[0, i] = de_test_temp.pval
        logfc[0, i] = de_test_temp.log_fold_change()
        if keep_full_test_objs:
            tests[0, i] = de_test_temp

    de_test = DifferentialExpressionTestVsRest(
        gene_ids=gene_names,
        pval=pvals,
        logfc=logfc,
        ave=np.mean(X, axis=0),
        groups=groups,
        tests=tests,
        correction_type=pval_correction
    )

    return de_test


def partition(
        data,
        partition: Union[str, np.ndarray, list],
        gene_names: str = None,
        sample_description: pd.DataFrame = None):
    r"""
    Perform differential expression test for each group. This class handles
    the partitioning of the data set, the differential test callls and
    the sumamry of the individual tests into one
    DifferentialExpressionTestMulti object. All functions the yield
    DifferentialExpressionTestSingle objects can be performed on each
    partition.

    Wraps _Partition so that doc strings are nice.

    :param data: input data
    :param gene_names: optional list/array of gene names which will be used if `data` does not implicitly store these
    :param sample_description: optional pandas.DataFrame containing sample annotations
    """
    return (_Partition(
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
        self.partition_idx = [np.where(self.partition == x)[0] for x in self.partitions]

    def two_sample(
            self,
            grouping: Union[str],
            test=None,
            noise_model: str = None,
            size_factors: np.ndarray = None,
            batch_size: int = None,
            training_strategy: Union[str, List[Dict[str, object]], Callable] = "AUTO",
            **kwargs
    ) -> _DifferentialExpressionTestMulti:
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
        for i, idx in enumerate(self.partition_idx):
            DETestsSingle.append(two_sample(
                data=self.X[idx, :],
                grouping=grouping,
                test=test,
                gene_names=self.gene_names,
                sample_description=self.sample_description.iloc[idx, :],
                noise_model=noise_model,
                size_factors=size_factors[idx] if size_factors is not None else None,
                batch_size=batch_size,
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
            grouping: Union[str],
            dtype="float32"
    ):
        """
        See annotation of de.test.t_test()

        :param grouping: str

            - column in data.obs/sample_description which contains the split of observations into the two groups.
        """
        DETestsSingle = []
        for i, idx in enumerate(self.partition_idx):
            DETestsSingle.append(t_test(
                data=self.X[idx, :],
                grouping=grouping,
                gene_names=self.gene_names,
                sample_description=self.sample_description.iloc[idx, :],
                dtype=dtype
            ))
        return DifferentialExpressionTestByPartition(
            partitions=self.partitions,
            tests=DETestsSingle,
            ave=np.mean(self.X, axis=0),
            correction_type="by_test")

    def wilcoxon(
            self,
            grouping: Union[str],
            dtype="float32"
    ):
        """
        See annotation of de.test.wilcoxon()

        :param grouping: str, array

            - column in data.obs/sample_description which contains the split of observations into the two groups.
            - array of length `num_observations` containing group labels
        """
        DETestsSingle = []
        for i, idx in enumerate(self.partition_idx):
            DETestsSingle.append(wilcoxon(
                data=self.X[idx, :],
                grouping=grouping,
                gene_names=self.gene_names,
                sample_description=self.sample_description.iloc[idx, :],
                dtype=dtype
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
            size_factors: np.ndarray = None,
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
        :param size_factors: 1D array of transformed library size factors for each cell in the 
            same order as in data
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
        for i, idx in enumerate(self.partition_idx):
            DETestsSingle.append(lrt(
                data=self.X[idx, :],
                reduced_formula=reduced_formula,
                full_formula=full_formula,
                reduced_formula_loc=reduced_formula_loc,
                full_formula_loc=full_formula_loc,
                reduced_formula_scale=reduced_formula_scale,
                full_formula_scale=full_formula_scale,
                gene_names=self.gene_names,
                sample_description=self.sample_description.iloc[idx, :],
                noise_model=noise_model,
                size_factors=size_factors[idx] if size_factors is not None else None,
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
            size_factors: np.ndarray = None,
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
        :param size_factors: 1D array of transformed library size factors for each cell in the 
            same order as in data
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
        for i, idx in enumerate(self.partition_idx):
            DETestsSingle.append(wald(
                data=self.X[idx, :],
                factor_loc_totest=factor_loc_totest,
                coef_to_test=coef_to_test,  # e.g. coef_to_test="B"
                formula=formula,
                formula_loc=formula_loc,
                formula_scale=formula_scale,
                gene_names=self.gene_names,
                sample_description=self.sample_description.iloc[idx, :],
                noise_model=noise_model,
                size_factors=size_factors[idx] if size_factors is not None else None,
                batch_size=batch_size,
                training_strategy=training_strategy,
                **kwargs
            ))
        return DifferentialExpressionTestByPartition(
            partitions=self.partitions,
            tests=DETestsSingle,
            ave=np.mean(self.X, axis=0),
            correction_type="by_test")
