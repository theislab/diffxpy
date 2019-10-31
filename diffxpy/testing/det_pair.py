import abc
try:
    import anndata
except ImportError:
    anndata = None
import batchglm.api as glm
import logging
import numpy as np
import pandas as pd
from typing import List, Union

from ..stats import stats
from . import correction

from .det import _DifferentialExpressionTestMulti

logger = logging.getLogger("diffxpy")


class _DifferentialExpressionTestPairwiseBase(_DifferentialExpressionTestMulti):
    """
    Pairwise differential expression tests base class.

    Defines API of accessing test results of pairs of groups. The underlying accessors depend
    on the type of test and are defined in the subclasses.
    """

    groups: List[str]
    _pval: Union[None, np.ndarray]
    _qval: Union[None, np.ndarray]
    _logfc: Union[None, np.ndarray]

    def _get_group_idx(self, groups0, groups1):
        if np.any([x not in self.groups for x in groups0]):
            raise ValueError('element of groups1 not recognized; not in groups %s' % self.groups)
        if np.any([x not in self.groups for x in groups1]):
            raise ValueError('element of groups2 not recognized; not in groups %s' % self.groups)

        return np.array([self.groups.index(x) for x in groups0]), \
               np.array([self.groups.index(x) for x in groups1])

    def _correction_pairs(self, idx0, idx1, method):
        if self._correction_type.lower() == "global":
            pval = self._pval_pairs(idx0=idx0, idx1=idx1)
            pval_reshape = np.reshape(pval, -1)
            qvals = correction.correct(pvals=pval_reshape, method=method)
            qvals = np.reshape(qvals, pval.shape)
        elif self._correction_type.lower() == "by_test":
            qvals = np.apply_along_axis(
                func1d=lambda pv: correction.correct(pvals=pv, method=method),
                axis=-1,
                arr=self._pval_pairs(idx0=idx0, idx1=idx1)
            )
        return qvals

    def log_fold_change(self, base=np.e, **kwargs):
        if base == np.e:
            return self._logfc
        else:
            return self._logfc / np.log(base)

    @abc.abstractmethod
    def _pval_pairs(self, idx0, idx1):
        """
        Test-specific p-values accessor for the comparison of groups1 to groups2.

        :param idx0: List of indices of first set of group of observations in pair-wise comparison.
        :param idx1: List of indices of second set of group of observations in pair-wise comparison.
        :return: p-values
        """
        pass

    def _qval_pairs(self, idx0, idx1, method="fdr_bh"):
        """
        Test-specific q-values accessor for the comparison of both sets of groups.

        :param idx0: List of indices of first set of group of observations in pair-wise comparison.
        :param idx1: List of indices of second set of group of observations in pair-wise comparison.
        :param method: Multiple testing correction method.
            Browse available methods in the annotation of statsmodels.stats.multitest.multipletests().
        :return: q-values
        """
        return self._correction_pairs(idx0=idx0, idx1=idx1, method=method)

    @abc.abstractmethod
    def _log_fold_change_pairs(self, idx0, idx1, base):
        """
        Test-specific log fold change-values accessor for the comparison of groups1 to groups2.

        :param idx0: List of indices of first set of group of observations in pair-wise comparison.
        :param idx1: List of indices of second set of group of observations in pair-wise comparison.
        :param base: Base of logarithm.
        :return: log fold change values
        """
        pass

    def pval_pairs(self, groups0, groups1):
        """
        Get p-values of the comparison of groups1 to groups2.

        :param groups0: List of identifier of first set of group of observations in pair-wise comparison.
        :param groups1: List of identifier of second set of group of observations in pair-wise comparison.
        :return: p-values
        """
        idx0, idx1 = self._get_group_idx(groups0=groups0, groups1=groups1)
        return self._pval_pairs(idx0=idx0, idx1=idx1)

    def qval_pairs(self, groups0, groups1, method="fdr_bh"):
        """
        Get q-values of the comparison of groups1 to groups2.

        :param groups0: List of identifier of first set of group of observations in pair-wise comparison.
        :param groups1: List of identifier of second set of group of observations in pair-wise comparison.
        :param method: Multiple testing correction method.
            Browse available methods in the annotation of statsmodels.stats.multitest.multipletests().
        :return: q-values
        """
        idx0, idx1 = self._get_group_idx(groups0=groups0, groups1=groups1)
        return self._qval_pairs(idx0=idx0, idx1=idx1, method=method)

    def log10_pval_pairs_clean(self, groups0, groups1, log10_threshold=-30):
        """
        Return log10 transformed and cleaned p-values.

        NaN p-values are set to one and p-values below log10_threshold
        in log10 space are set to log10_threshold.

        :param groups0: List of identifier of first set of group of observations in pair-wise comparison.
        :param groups1: List of identifier of second set of group of observations in pair-wise comparison.
        :param log10_threshold: minimal log10 p-value to return.
        :return: Cleaned log10 transformed p-values.
        """
        pvals = np.reshape(self.pval_pairs(groups0=groups0, groups1=groups1), -1)
        pvals = np.nextafter(0, 1, out=pvals, where=pvals == 0)
        log10_pval_clean = np.log(pvals) / np.log(10)
        log10_pval_clean[np.isnan(log10_pval_clean)] = 1
        log10_pval_clean = np.clip(log10_pval_clean, log10_threshold, 0, log10_pval_clean)
        return log10_pval_clean

    def log10_qval_pairs_clean(self, groups0, groups1, log10_threshold=-30):
        """
        Return log10 transformed and cleaned q-values.

        NaN p-values are set to one and q-values below log10_threshold
        in log10 space are set to log10_threshold.

        :param groups0: List of identifier of first set of group of observations in pair-wise comparison.
        :param groups1: List of identifier of second set of group of observations in pair-wise comparison.
        :param log10_threshold: minimal log10 q-value to return.
        :return: Cleaned log10 transformed q-values.
        """
        qvals = np.reshape(self.qval_pairs(groups0=groups0, groups1=groups1), -1)
        qvals = np.nextafter(0, 1, out=qvals, where=qvals == 0)
        log10_qval_clean = np.log(qvals) / np.log(10)
        log10_qval_clean[np.isnan(log10_qval_clean)] = 1
        log10_qval_clean = np.clip(log10_qval_clean, log10_threshold, 0, log10_qval_clean)
        return log10_qval_clean

    def log_fold_change_pairs(
            self,
            groups0,
            groups1,
            base=np.e
    ):
        """
        Get log fold changes of the comparison of group1 and group2.

        :param groups0: List of identifier of first set of group of observations in pair-wise comparison.
        :param groups1: List of identifier of second set of group of observations in pair-wise comparison.
        :param base: Base of logarithm.
        :return: log fold changes
        """
        idx0, idx1 = self._get_group_idx(groups0=groups0, groups1=groups1)
        return self._log_fold_change_pairs(idx0=idx0, idx1=idx1, base=base)

    def summary(
            self,
            qval_thres=None,
            fc_upper_thres=None,
            fc_lower_thres=None,
            mean_thres=None,
            **kwargs
    ) -> pd.DataFrame:
        """
        Summarize differential expression results into an output table.

        :param qval_thres: Upper bound of corrected p-values for gene to be included.
        :param fc_upper_thres: Upper bound of fold-change for gene to be included.
        :param fc_lower_thres: Lower bound of fold-change p-values for gene to be included.
        :param mean_thres: Lower bound of average expression for gene to be included.
        :return: Summary table of differential expression test.
        """
        res = super(_DifferentialExpressionTestMulti, self).summary(**kwargs)

        return self._threshold_summary(
            res=res,
            qval_thres=qval_thres,
            fc_upper_thres=fc_upper_thres,
            fc_lower_thres=fc_lower_thres,
            mean_thres=mean_thres
        )

    def summary_pairs(
            self,
            groups0,
            groups1,
            qval_thres=None,
            fc_upper_thres=None,
            fc_lower_thres=None,
            mean_thres=None,
            method="fdr_bh"
    ) -> pd.DataFrame:
        """
        Summarize differential expression results into an output table.

        :param groups0: List of identifier of first set of group of observations in pair-wise comparison.
        :param groups1: List of identifier of second set of group of observations in pair-wise comparison.
        :param qval_thres: Upper bound of corrected p-values for gene to be included.
        :param fc_upper_thres: Upper bound of fold-change for gene to be included.
        :param fc_lower_thres: Lower bound of fold-change p-values for gene to be included.
        :param mean_thres: Lower bound of average expression for gene to be included.
        :param method: Multiple testing correction method.
            Browse available methods in the annotation of statsmodels.stats.multitest.multipletests().
        :return: pandas.DataFrame with the following columns:
            - gene: the gene id's
            - pval: the minimum per-gene p-value of all tests
            - qval: the minimum per-gene q-value of all tests
            - log2fc: the maximal/minimal (depending on which one is higher) log2 fold change of the genes
            - mean: the mean expression of the gene across all groups
        """
        assert self.gene_ids is not None

        pval = self.pval_pairs(groups0=groups0, groups1=groups1)
        qval = self.qval_pairs(groups0=groups0, groups1=groups1, method=method)

        # calculate maximum logFC of lower triangular fold change matrix
        raw_logfc = self.log_fold_change_pairs(groups0=groups0, groups1=groups1, base=2)

        # first flatten all dimensions up to the last 'gene' dimension
        flat_logfc = raw_logfc.reshape(-1, raw_logfc.shape[-1])
        # next, get argmax of flattened logfc and unravel the true indices from it
        r, c = np.unravel_index(flat_logfc.argmax(0), raw_logfc.shape[:2])
        # if logfc is maximal in the lower triangular matrix, multiply it with -1
        logfc = raw_logfc[r, c, np.arange(raw_logfc.shape[-1])] * np.where(r <= c, 1, -1)

        res = pd.DataFrame({
            "gene": self.gene_ids,
            "pval": np.min(pval, axis=(0, 1)),
            "qval": np.min(qval, axis=(0, 1)),
            "log2fc": np.asarray(logfc),
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


class DifferentialExpressionTestPairwiseStandard(_DifferentialExpressionTestPairwiseBase):
    """
    Pairwise differential expression tests class for tests other than z-tests.
    """

    def __init__(
            self,
            gene_ids,
            pval,
            logfc,
            ave,
            groups,
            tests,
            correction_type: str
    ):
        super().__init__(correction_type=correction_type)
        self._gene_ids = np.asarray(gene_ids)
        self._logfc = logfc
        self._pval = pval
        self._mean = np.asarray(ave).flatten()
        self.groups = list(np.asarray(groups))
        self._tests = tests

        _ = self.qval

    @property
    def gene_ids(self) -> np.ndarray:
        return self._gene_ids

    @property
    def x(self):
        return None

    @property
    def tests(self):
        """
        If `keep_full_test_objs` was set to `True`, this will return a matrix of differential expression tests.
        """
        if self._tests is None:
            raise ValueError("Individual tests were not kept!")

        return self._tests

    def _pval_pairs(self, idx0, idx1):
        """
        Test-specific p-values accessor for the comparison of groups1 to groups2.

        :param idx0: List of indices of first set of group of observations in pair-wise comparison.
        :param idx1: List of indices of second set of group of observations in pair-wise comparison.
        :return: p-values
        """
        assert np.all([x < self._pval.shape[1] for x in idx0])
        assert np.all([x < self._pval.shape[1] for x in idx1])
        return self._pval[idx0, :, :][:, idx1, :]

    def _log_fold_change_pairs(self, idx0, idx1, base):
        """
        Test-specific log fold change-values accessor for the comparison of groups1 to groups2.

        :param idx0: List of indices of first set of group of observations in pair-wise comparison.
        :param idx1: List of indices of second set of group of observations in pair-wise comparison.
        :param base: Base of logarithm.
        :return: log fold change values
        """
        assert np.all([x < self._pval.shape[1] for x in idx0])
        assert np.all([x < self._pval.shape[1] for x in idx1])
        if base == np.e:
            return self._logfc[idx0, :, :][:, idx1, :]
        else:
            return self._logfc[idx0, :, :][:, idx1, :] / np.log(base)


class DifferentialExpressionTestZTest(_DifferentialExpressionTestPairwiseBase):
    """
    Pairwise unit_test between more than 2 groups per gene. This close harbors tests that are precomputed
    across all pairs of groups. DifferentialExpressionTestZTestLazy is the alternative class that only allows
    lazy test evaluation.
    """

    model_estim: glm.typing.EstimatorBaseTyping
    theta_mle: np.ndarray
    theta_sd: np.ndarray

    def __init__(
            self,
            model_estim: glm.typing.EstimatorBaseTyping,
            grouping,
            groups,
            correction_type: str
    ):
        super().__init__(correction_type=correction_type)
        self.model_estim = model_estim
        self.grouping = grouping
        self.groups = list(np.asarray(groups))

        # Values of parameter estimates: coefficients x genes array with one coefficient per group
        self._theta_mle = model_estim.a_var
        # Standard deviation of estimates: coefficients x genes array with one coefficient per group
        # Need .copy() here as nextafter needs mutabls copy.
        theta_sd = np.diagonal(model_estim.fisher_inv, axis1=-2, axis2=-1).T.copy()
        theta_sd = np.nextafter(0, np.inf, out=theta_sd, where=theta_sd < np.nextafter(0, np.inf))
        self._theta_sd = np.sqrt(theta_sd)
        self._logfc = None

        # Call tests in constructor.
        _ = self.pval
        _ = self.qval

    def _test(self, **kwargs):
        pvals = np.tile(np.NaN, [len(self.groups), len(self.groups), self.model_estim.x.shape[1]])
        for i, g1 in enumerate(self.groups):
            for j, g2 in enumerate(self.groups[(i + 1):]):
                j = j + i + 1
                pvals[i, j, :] = stats.two_coef_z_test(
                    theta_mle0=self._theta_mle[i, :],
                    theta_mle1=self._theta_mle[j, :],
                    theta_sd0=self._theta_sd[i, :],
                    theta_sd1=self._theta_sd[j, :]
                )
                pvals[j, i, :] = pvals[i, j, :]

        return pvals

    @property
    def gene_ids(self) -> np.ndarray:
        return np.asarray(self.model_estim.input_data.features)

    @property
    def x(self):
        return self.model_estim.x

    @property
    def log_likelihood(self):
        return self.model_estim.log_likelihood

    @property
    def model_gradient(self):
        return self.model_estim.jacobian

    def _ave(self):
        """
        Returns the mean expression by gene

        :return: np.ndarray
        """

        return np.asarray(np.mean(self.model_estim.x, axis=0)).flatten()

    def _pval_pairs(self, idx0, idx1):
        """
        Test-specific p-values accessor for the comparison of groups1 to groups2.

        :param idx0: List of indices of first set of group of observations in pair-wise comparison.
        :param idx1: List of indices of second set of group of observations in pair-wise comparison.
        :return: p-values
        """
        assert np.all([x < self.pval.shape[1] for x in idx0])
        assert np.all([x < self.pval.shape[1] for x in idx1])
        return self.pval[idx0, :, :][:, idx1, :]

    def _log_fold_change_pairs(self, idx0, idx1, base):
        """
        Test-specific log fold change-values accessor for the comparison of groups1 to groups2.

        :param idx0: List of indices of first set of group of observations in pair-wise comparison.
        :param idx1: List of indices of second set of group of observations in pair-wise comparison.
        :param base: Base of logarithm.
        :return: log fold change values
        """
        logfc = np.tile(np.NaN, [len(idx0), len(idx1), self.model_estim.x.shape[1]])
        for i, xi in enumerate(idx0):
            for j, xj in enumerate(idx1):
                logfc[i, j, :] = self._theta_mle[xj, :] - self._theta_mle[xi, :]
                logfc[j, i, :] = -logfc[i, j, :]

        if base == np.e:
            return logfc
        else:
            return logfc / np.log(base)


class _DifferentialExpressionTestPairwiseLazyBase(_DifferentialExpressionTestPairwiseBase):
    """
    Lazy pairwise differential expression tests base class.

    In addition to the method of a standard pairwise test, this base class throws errors for attributes
    that are not accessible in a lazy pairwise test.
    """

    def _test(self, **kwargs):
        """
        This function is not available in lazy results evaluation as it would
        require all pairwise tests to be performed.
        """
        raise ValueError("This function is not available in lazy results evaluation as it would "
                         "require all pairwise tests to be performed.")

    @property
    def pval(self, **kwargs):
        """
        This function is not available in lazy results evaluation as it would
        require all pairwise tests to be performed.
        """
        raise ValueError("This function is not available in lazy results evaluation as it would "
                         "require all pairwise tests to be performed.")

    @property
    def qval(self, **kwargs):
        """
        This function is not available in lazy results evaluation as it would
        require all pairwise tests to be performed.
        """
        raise ValueError("This function is not available in lazy results evaluation as it would "
                         "require all pairwise tests to be performed.")

    def log_fold_change(self, base=np.e, **kwargs):
        """
        This function is not available in lazy results evaluation as it would
        require all pairwise tests to be performed.
        """
        raise ValueError("This function is not available in lazy results evaluation as it would "
                         "require all pairwise tests to be performed.")

    @property
    def summary(self, **kwargs):
        """
        This function is not available in lazy results evaluation as it would
        require all pairwise tests to be performed.
        """
        raise ValueError("This function is not available in lazy results evaluation as it would "
                         "require all pairwise tests to be performed.")

    @abc.abstractmethod
    def _test_pairs(self, idx0, idx1):
        """
        Run differential expression tests on selected pairs of groups.

        :param idx0: List of indices of first set of group of observations in pair-wise comparison.
        :param idx1: List of indices of second set of group of observations in pair-wise comparison.
        :return: p-values
        """
        pass

    def _pval_pairs(self, idx0, idx1):
        """
        Test-specific p-values accessor for the comparison of groups0 to groups1.

        :param idx0: List of indices of first set of group of observations in pair-wise comparison.
        :param idx1: List of indices of second set of group of observations in pair-wise comparison.
        :return: p-values
        """
        return self._test_pairs(idx0=idx0, idx1=idx1)


class DifferentialExpressionTestZTestLazy(_DifferentialExpressionTestPairwiseLazyBase):
    """
    Pairwise unit_test between more than 2 groups per gene with lazy evaluation.

    This class performs pairwise tests upon enquiry only and does not store them
    and is therefore suited so very large group sets for which the lfc and
    p-value matrices of the size [genes, groups, groups] are too big to fit into
    memory.
    """

    model_estim: glm.typing.EstimatorBaseTyping
    _theta_mle: np.ndarray
    _theta_sd: np.ndarray

    def __init__(
            self,
            model_estim: glm.typing.EstimatorBaseTyping,
            grouping, groups,
            correction_type="global"
    ):
        _DifferentialExpressionTestMulti.__init__(
            self=self,
            correction_type=correction_type
        )
        self.model_estim = model_estim
        self.grouping = grouping
        if isinstance(groups, list):
            self.groups = groups
        else:
            self.groups = groups.tolist()

        # Values of parameter estimates: coefficients x genes array with one coefficient per group
        self._theta_mle = model_estim.a_var
        # Standard deviation of estimates: coefficients x genes array with one coefficient per group
        # Need .copy() here as nextafter needs mutabls copy.
        theta_sd = np.diagonal(model_estim.fisher_inv, axis1=-2, axis2=-1).T.copy()
        theta_sd = np.nextafter(0, np.inf, out=theta_sd, where=theta_sd < np.nextafter(0, np.inf))
        self._theta_sd = np.sqrt(theta_sd)

    def _test_pairs(self, idx0, idx1):
        """
        Run differential expression tests on selected pairs of groups.

        :param idx0: List of indices of first set of group of observations in pair-wise comparison.
        :param idx1: List of indices of second set of group of observations in pair-wise comparison.
        :return: p-values
        """
        pvals = np.tile(np.NaN, [len(idx0), len(idx1), self.model_estim.x.shape[1]])
        for i, xi in enumerate(idx0):
            for j, xj in enumerate(idx1):
                if i != j:
                    pvals[i, j, :] = stats.two_coef_z_test(
                        theta_mle0=self._theta_mle[xi, :],
                        theta_mle1=self._theta_mle[xj, :],
                        theta_sd0=self._theta_sd[xi, :],
                        theta_sd1=self._theta_sd[xj, :]
                    )
                else:
                    pvals[i, j, :] = np.array([1.])

        return pvals

    @property
    def gene_ids(self) -> np.ndarray:
        return np.asarray(self.model_estim.input_data.features)

    @property
    def x(self):
        return self.model_estim.x

    @property
    def log_likelihood(self):
        return self.model_estim.log_likelihood

    @property
    def model_gradient(self):
        return self.model_estim.jacobian

    def _ave(self):
        """
        Returns the mean expression by gene

        :return: np.ndarray
        """
        return np.asarray(np.mean(self.model_estim.x, axis=0)).flatten()

    def _log_fold_change_pairs(self, idx0, idx1, base):
        """
        Test-specific log fold change-values accessor for the comparison of both sets of groups.

        :param idx0: List of indices of first set of group of observations in pair-wise comparison.
        :param idx1: List of indices of second set of group of observations in pair-wise comparison.
        :param base: Base of logarithm.
        :return: log fold change values
        """
        logfc = np.zeros(shape=(len(idx0), len(idx1), self._theta_mle.shape[1]))
        for i, xi in enumerate(idx0):
            for j, xj in enumerate(idx1):
                logfc[i, j, :] = self._theta_mle[xi, :] - self._theta_mle[xj, :]

        if base == np.e:
            return logfc
        else:
            return logfc / np.log(base)
