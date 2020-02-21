import abc
try:
    import anndata
except ImportError:
    anndata = None
import batchglm.api as glm
import dask
import logging
import numpy as np
import pandas as pd
import scipy
import scipy.sparse
import sparse
from typing import Union

from .det import _DifferentialExpressionTestSingle, DifferentialExpressionTestWald, DifferentialExpressionTestLRT

logger = logging.getLogger("diffxpy")


class _DifferentialExpressionTestCont(_DifferentialExpressionTestSingle):
    _de_test: _DifferentialExpressionTestSingle
    _model_estim: glm.typing.EstimatorBaseTyping
    _size_factors: np.ndarray
    _continuous_coords: np.ndarray
    _spline_coefs: list

    def __init__(
            self,
            de_test: _DifferentialExpressionTestSingle,
            model_estim: glm.typing.EstimatorBaseTyping,
            size_factors: np.ndarray,
            continuous_coords: np.ndarray,
            spline_coefs: list,
            interpolated_spline_basis: np.ndarray,
            noise_model: str
    ):
        self._de_test = de_test
        self._model_estim = model_estim
        self._size_factors = size_factors
        self._continuous_coords = continuous_coords
        self._spline_coefs = spline_coefs
        self._interpolated_spline_basis = interpolated_spline_basis
        self.noise_model = noise_model

    @property
    def gene_ids(self) -> np.ndarray:
        return self._de_test.gene_ids

    @property
    def x(self):
        return self._de_test.x

    @property
    def pval(self) -> np.ndarray:
        return self._de_test.pval

    @property
    def qval(self) -> np.ndarray:
        return self._de_test.qval

    @property
    def mean(self) -> np.ndarray:
        return self._de_test.mean

    @property
    def log_likelihood(self) -> np.ndarray:
        return self._de_test.log_likelihood

    def summary(
            self,
            non_numeric=False,
            qval_thres=None,
            fc_upper_thres=None,
            fc_lower_thres=None,
            mean_thres=None
    ) -> pd.DataFrame:
        """
        Summarize differential expression results into an output table.

        :param non_numeric: Whether to include non-numeric covariates in fit.
        """
        # Collect summary from differential test object.
        res = self._de_test.summary()
        # Overwrite fold change with fold change from temporal model.
        # Note that log2_fold_change calls log_fold_change from this class
        # and not from the self._de_test object,
        # which is called by self._de_test.summary().
        res['log2fc'] = self.log2_fold_change()

        res = self._threshold_summary(
            res=res,
            qval_thres=qval_thres,
            fc_upper_thres=fc_upper_thres,
            fc_lower_thres=fc_lower_thres,
            mean_thres=mean_thres
        )

        return res

    def log_fold_change(self, base=np.e, genes=None, non_numeric=False):
        """
        Return log_fold_change based on fitted expression values by gene.

        The log_fold_change is defined as the log of the fold change
        from the minimal to the maximal fitted value by gene.

        :param base: Basis of logarithm.
        :param genes: Genes for which to return maximum fitted value. Defaults
            to all genes if None.
        :param non_numeric: Whether to include non-numeric covariates in fit.
        :return: Log-fold change of fitted expression value by gene.
        """
        if genes is None:
            idx = np.arange(0, self.x.shape[1])
        else:
            idx, genes = self._idx_genes(genes)

        min_val, max_val = self.min_max(genes=idx, non_numeric=non_numeric)
        max_val = np.nextafter(0, 1, out=max_val, where=max_val == 0)
        min_val = np.nextafter(0, 1, out=min_val, where=min_val == 0)
        return (np.log(max_val) - np.log(min_val)) / np.log(base)

    def log2_fold_change(self, genes=None, non_numeric=False):
        """
        Calculates the pairwise log_2 fold change(s) for this DifferentialExpressionTest.

        :param genes: Genes for which to return maximum fitted value. Defaults
            to all genes if None.
        :param non_numeric: Whether to include non-numeric covariates in fit.
        :return: Log-fold change of fitted expression value by gene.
        """
        return self.log_fold_change(base=2, genes=genes, non_numeric=non_numeric)

    def log10_fold_change(self, genes=None, non_numeric=False):
        """
        Calculates the log_10 fold change(s) for this DifferentialExpressionTest.

        :param genes: Genes for which to return maximum fitted value. Defaults
            to all genes if None.
        :param non_numeric: Whether to include non-numeric covariates in fit.
        :return: Log-fold change of fitted expression value by gene.
        """
        return self.log_fold_change(base=10, genes=genes, non_numeric=non_numeric)

    def _filter_genes_str(self, genes: list):
        """
        Filter genes indexed by ID strings by list of genes given in data set.

        :param genes: List of genes to filter.
        :return: Filtered list of genes
        """
        genes_found = np.array([x in self.gene_ids for x in genes])
        if any(np.logical_not(genes_found)):
            logger.info("did not find some genes, omitting")
            genes = genes[genes_found]
        return genes

    def _filter_genes_int(self, genes: list):
        """
        Filter genes indexed by integers by gene list length.

        :param genes: List of genes to filter.
        :return: Filtered list of genes
        """
        genes_found = np.array([x < self.x.shape[1] for x in genes])
        if any(np.logical_not(genes_found)):
            logger.info("did not find some genes, omitting")
            genes = genes[genes_found]
        return genes

    def _idx_genes(self, genes):
        if not isinstance(genes, list):
            if isinstance(genes, np.ndarray):
                genes = genes.tolist()
            elif isinstance(genes, pd.Index):
                genes = genes.tolist()
            else:
                genes = [genes]

        if isinstance(genes[0], str):
            genes = self._filter_genes_str(genes)
            idx = np.array([self.gene_ids.tolist().index(x) for x in genes])
        elif isinstance(genes[0], int) or isinstance(genes[0], np.number):
            genes = self._filter_genes_int(genes)
            idx = genes
            genes = [self.gene_ids.tolist()[x] for x in idx]
        else:
            raise ValueError("only string and integer elements allowed in genes")
        return idx, genes

    def _spline_par_loc_idx(self, intercept=True):
        """
        Get indices of spline basis model parameters in
        entire location parameter model parameter set.

        :param intercept: Whether to include intercept.
        :return: Indices of spline basis parameters of location model.
        """
        par_loc_names = self._model_estim.input_data.loc_names
        idx = [par_loc_names.index(x) for x in self._spline_coefs]
        if 'Intercept' in par_loc_names and intercept:
            idx = np.concatenate([np.where([[x == 'Intercept' for x in par_loc_names]])[0], idx])
        return idx

    def _continuous_model(self, idx, non_numeric=False):
        """
        Recover continuous fit for a gene in observed time points.

        Gives interpolation for each observation.

        :param idx: Index of genes to recover fit for.
        :param non_numeric: Whether to include non-numeric covariates in fit.
        :return: Continuuos fit for each cell for given gene.
        """
        idx = np.asarray(idx)
        if len(idx.shape) == 0:
            idx = np.array([idx])

        if non_numeric:
            mu = np.matmul(self._model_estim.input_data.design_loc,
                           self._model_estim.model.a[:, idx])
            if self._size_factors is not None:
                mu = mu + self._model_estim.input_data.size_factors
        else:
            idx_basis = self._spline_par_loc_idx(intercept=True)
            mu = np.matmul(self._model_estim.input_data.design_loc[:, idx_basis],
                           self._model_estim.model.a[idx_basis, :][:, idx])
        if isinstance(mu, dask.array.core.Array):
            mu = mu.compute()

        mu = np.exp(mu)
        return mu

    def _continuous_interpolation(self, idx):
        """
        Recover continuous fit for a gene in uniformely spaced time grid.

        Gives interpolation for each grid point - more efficient for plotting for example.

        :param idx: Index of genes to recover fit for.
        :return: Continuuos fit for each cell for given gene.
        """
        idx = np.asarray(idx)
        if len(idx.shape) == 0:
            idx = np.array([idx])

        idx_basis = self._spline_par_loc_idx(intercept=True)
        a = self._model_estim.model.a[idx_basis, :]
        if isinstance(a, dask.array.core.Array):
            a = a.compute()[:, idx]
        else:
            a = a[:, idx]
        eta_loc = np.matmul(self._interpolated_spline_basis[:, :-1], a)
        mu = np.exp(eta_loc)
        t_eval = self._interpolated_spline_basis[:, -1]
        return t_eval, mu

    def min_max(self, genes, non_numeric=False):
        """
        Return maximum and minimum of fitted expression value by gene.

        :param genes: Genes for which to return maximum fitted value.
        :param non_numeric: Whether to include non-numeric covariates in fit.
        :return: Array of minimum and maximum fitted expression values by gene.
        """
        idx, genes = self._idx_genes(genes)
        mins = []
        maxs = []
        for i in idx:
            vals = self._continuous_model(idx=i, non_numeric=non_numeric)
            mins.append(np.min(vals))
            maxs.append(np.max(vals))
        return np.array(mins), np.array(maxs)

    def max(self, genes, non_numeric=False):
        """
        Return maximum fitted expression value by gene.

        :param genes: Genes for which to return maximum fitted value.
        :param non_numeric: Whether to include non-numeric covariates in fit.
        :return: Maximum fitted expression value by gene.
        """
        idx, genes = self._idx_genes(genes)
        return np.array([np.max(self._continuous_model(idx=i, non_numeric=non_numeric))
                         for i in idx])

    def min(self, genes, non_numeric=False):
        """
        Return minimum fitted expression value by gene.

        :param genes: Genes for which to return maximum fitted value.
        :param non_numeric: Whether to include non-numeric covariates in fit.
        :return: Maximum fitted expression value by gene.
        """
        idx, genes = self._idx_genes(genes)
        return np.array([np.min(self._continuous_model(idx=i, non_numeric=non_numeric))
                         for i in idx])

    def argmax(self, genes, non_numeric=False):
        """
        Return maximum fitted expression value by gene.

        :param genes: Genes for which to return maximum fitted value.
        :param non_numeric: Whether to include non-numeric covariates in fit.
        :return: Maximum fitted expression value by gene.
        """
        idx, genes = self._idx_genes(genes)
        idx_cont = np.array([np.argmax(self._continuous_model(idx=i, non_numeric=non_numeric))
                             for i in idx])
        return self._continuous_coords[idx_cont]

    def argmin(self, genes, non_numeric=False):
        """
        Return minimum fitted expression value by gene.

        :param genes: Genes for which to return maximum fitted value.
        :param non_numeric: Whether to include non-numeric covariates in fit.
        :return: Maximum fitted expression value by gene.
        """
        idx, genes = self._idx_genes(genes)
        idx_cont = np.array([np.argmin(self._continuous_model(idx=i, non_numeric=non_numeric))
                             for i in idx])
        return self._continuous_coords[idx_cont]

    def plot_genes(
            self,
            genes,
            hue=None,
            scalings=None,
            size=1,
            log=True,
            save=None,
            show=True,
            ncols=2,
            row_gap=0.3,
            col_gap=0.25,
            return_axs=False
    ):
        """
        Plot observed data and spline fits of selected genes.

        :param genes: Gene IDs to plot.
        :param hue: Confounder to include in plot. Must be length number of observations.
        :param scalings: Names of scaling coefficients to plot separate model curves for.
        :param size: Point size.
        :param log: Whether to log values.
        :param save: Path+file name stem to save plots to.
            File will be save+"_genes.png". Does not save if save is None.
        :param show: Whether to display plot.
        :param ncols: Number of columns in plot grid if multiple genes are plotted.
        :param row_gap: Vertical gap between panel rows relative to panel height.
        :param col_gap: Horizontal gap between panel columns relative to panel width.
        :param return_axs: Whether to return axis objects of plots.
        :return: Matplotlib axis objects.
        """

        import seaborn as sns
        import matplotlib.pyplot as plt
        from matplotlib import gridspec
        from matplotlib import rcParams

        plt.ioff()

        gene_idx, gene_ids = self._idx_genes(genes)

        # Set up gridspec.
        ncols = ncols if len(gene_idx) > ncols else len(gene_idx)
        nrows = len(gene_idx) // ncols + (len(gene_idx) - (len(gene_idx) // ncols) * ncols)
        gs = gridspec.GridSpec(
            nrows=nrows,
            ncols=ncols,
            hspace=row_gap,
            wspace=col_gap
        )

        # Define figure size based on panel number and grid.
        fig = plt.figure(
            figsize=(
                ncols * rcParams['figure.figsize'][0],  # width in inches
                nrows * rcParams['figure.figsize'][1] * (1 + row_gap)  # height in inches
            )
        )

        # Build axis objects in loop.
        axs = []
        for i, g in enumerate(gene_idx):
            ax = plt.subplot(gs[i])
            axs.append(ax)

            y = self.x[:, g]
            if isinstance(y, dask.array.core.Array):
                y = y.compute()
            if isinstance(y, scipy.sparse.spmatrix) or isinstance(y, sparse.COO):
                y = np.asarray(y.todense()).flatten()
                if self._model_estim.input_data.size_factors is not None:
                    y = y / self._model_estim.input_data.size_factors
            t_continuous, yhat = self._continuous_interpolation(idx=g)
            yhat = yhat.flatten()
            if scalings is not None:
                yhat = np.vstack([
                    [yhat],
                    [
                        yhat * np.expand_dims(
                            np.exp(self._model_estim.a_var[self._model_estim.input_data.loc_names.index(x), g]),
                            axis=0
                        )
                        for i, x in enumerate(scalings)
                    ]
                ])
            else:
                yhat = np.expand_dims(yhat, axis=0)
            if log:
                y = np.log(y + 1)
                yhat = np.log(yhat + 1)

            if isinstance(yhat, dask.array.core.Array):
                yhat = yhat.compute()

            sns.scatterplot(
                x=self._continuous_coords,
                y=y,
                hue=hue,
                size=size,
                ax=ax,
                legend=False
            )
            for j in range(yhat.shape[0]):
                sns.lineplot(
                    x=t_continuous,
                    y=yhat[j, :],
                    hue=None,
                    ax=ax
                )

            ax.set_title(gene_ids[i])
            ax.set_xlabel("continuous")
            if log:
                ax.set_ylabel("log expression")
            else:
                ax.set_ylabel("expression")

        # Save, show and return figure.
        if save is not None:
            plt.savefig(save+'_genes.png')

        if show:
            plt.show()

        plt.close(fig)
        plt.ion()

        if return_axs:
            return axs
        else:
            return

    def plot_heatmap(
            self,
            genes: Union[list, np.ndarray],
            save=None,
            show=True,
            transform: str = "zscore",
            nticks=10,
            cmap: str = "YlGnBu",
            width=10,
            height_per_gene=0.5,
            return_axs=False
    ):
        """
        Plot observed data and spline fits of selected genes.

        :param genes: Gene IDs to plot.
        :param save: Path+file name stem to save plots to.
            File will be save+"_genes.png". Does not save if save is None.
        :param show: Whether to display plot.
        :param transform: Gene-wise transform to use.
        :param nticks: Number of x ticks.
        :param cmap: matplotlib cmap.
        :param width: Width of heatmap figure.
        :param height_per_gene: Height of each row (gene) in heatmap figure.
        :param return_axs: Whether to return axis objects of plots.
        :return: Matplotlib axis objects.
        """
        import seaborn as sns
        import matplotlib.pyplot as plt

        plt.ioff()

        gene_idx, gene_ids = self._idx_genes(genes)

        # Define figure.
        fig = plt.figure(figsize=(width, height_per_gene * len(gene_idx)))
        ax = fig.add_subplot(111)

        # Build heatmap matrix.
        # Add in data.
        xcoord, data = self._continuous_interpolation(idx=gene_idx)
        if isinstance(data, dask.array.core.Array):
            data = data.compute()
        data = data.T

        if transform.lower() == "log10":
            data = np.nextafter(0, 1, out=data, where=data == 0)
            data = np.log(data) / np.log(10)
        elif transform.lower() == "zscore":
            mu = np.mean(data, axis=0, keepdims=True)
            sd = np.std(data, axis=0, keepdims=True)
            sd = np.nextafter(0, 1, out=sd, where=sd == 0)
            data = (data - mu) / sd
        elif transform.lower() == "none":
            pass
        else:
            raise ValueError("transform not recognized in plot_heatmap()")

        # Create heatmap.
        sns.heatmap(data=data, cmap=cmap, ax=ax)

        # Set up axis labels.
        xtick_pos = np.asarray(np.round(np.linspace(
            start=0,
            stop=data.shape[1] - 1,
            num=nticks,
            endpoint=True
        )), dtype=int)
        xtick_lab = [str(np.round(xcoord[np.argmin(np.abs(xcoord - xcoord[i]))], 2))
                     for i in xtick_pos]
        ax.set_xticks(xtick_pos)
        ax.set_xticklabels(xtick_lab)
        ax.set_xlabel("continuous")
        plt.yticks(np.arange(len(genes)), gene_ids, rotation='horizontal')
        ax.set_ylabel("genes")

        # Save, show and return figure.
        if save is not None:
            plt.savefig(save + '_genes.png')

        if show:
            plt.show()

        plt.close(fig)
        plt.ion()

        if return_axs:
            return ax
        else:
            return


class DifferentialExpressionTestWaldCont(_DifferentialExpressionTestCont):
    de_test: DifferentialExpressionTestWald

    def __init__(
            self,
            de_test: DifferentialExpressionTestWald,
            size_factors: np.ndarray,
            continuous_coords: np.ndarray,
            spline_coefs: list,
            interpolated_spline_basis: np.ndarray,
            noise_model: str
    ):
        super(DifferentialExpressionTestWaldCont, self).__init__(
            de_test=de_test,
            model_estim=de_test.model_estim,
            size_factors=size_factors,
            continuous_coords=continuous_coords,
            spline_coefs=spline_coefs,
            interpolated_spline_basis=interpolated_spline_basis,
            noise_model=noise_model
        )


class DifferentialExpressionTestLRTCont(_DifferentialExpressionTestCont):
    de_test: DifferentialExpressionTestLRT

    def __init__(
            self,
            de_test: DifferentialExpressionTestLRT,
            size_factors: np.ndarray,
            continuous_coords: np.ndarray,
            spline_coefs: list,
            interpolated_spline_basis: np.ndarray,
            noise_model: str
    ):
        super().__init__(
            de_test=de_test,
            model_estim=de_test.full_estim,
            size_factors=size_factors,
            continuous_coords=continuous_coords,
            spline_coefs=spline_coefs,
            interpolated_spline_basis=interpolated_spline_basis,
            noise_model=noise_model
        )
