import anndata
try:
    from anndata.base import Raw
except ImportError:
    from anndata import Raw
import batchglm.api as glm
import logging
import numpy as np
import pandas as pd
import patsy
import scipy.sparse
from typing import Union, List, Dict, Callable, Tuple

from .external import _fit
from .external import parse_gene_names, parse_sample_description, parse_size_factors, parse_grouping, \
    constraint_system_from_star


def model(
        data: Union[anndata.AnnData, Raw, np.ndarray, scipy.sparse.csr_matrix, glm.typing.InputDataBase],
        formula_loc: Union[None, str] = None,
        formula_scale: Union[None, str] = "~1",
        as_numeric: Union[List[str], Tuple[str], str] = (),
        init_a: Union[np.ndarray, str] = "AUTO",
        init_b: Union[np.ndarray, str] = "AUTO",
        gene_names: Union[np.ndarray, list] = None,
        sample_description: Union[None, pd.DataFrame] = None,
        dmat_loc: Union[patsy.design_info.DesignMatrix] = None,
        dmat_scale: Union[patsy.design_info.DesignMatrix] = None,
        constraints_loc: Union[None, List[str], Tuple[str, str], dict, np.ndarray] = None,
        constraints_scale: Union[None, List[str], Tuple[str, str], dict, np.ndarray] = None,
        noise_model: str = "nb",
        size_factors: Union[np.ndarray, pd.core.series.Series, str] = None,
        batch_size: int = None,
        training_strategy: Union[str, List[Dict[str, object]], Callable] = "AUTO",
        quick_scale: bool = False,
        dtype="float64",
        **kwargs
):
    """
    Fit model via maximum likelihood for each gene.

    :param data: Input data matrix (observations x features) or (cells x genes).
    :param formula_loc: formula
        model formula for location and scale parameter models.
        If not specified, `formula` will be used instead.
    :param formula_scale: formula
        model formula for scale parameter model.
        If not specified, `formula` will be used instead.
    :param as_numeric:
        Which columns of sample_description to treat as numeric and
        not as categorical. This yields columns in the design matrix
        which do not correspond to one-hot encoded discrete factors.
        This makes sense for number of genes, time, pseudotime or space
        for example.
    :param init_a: (Optional) Low-level initial values for a.
        Can be:

        - str:
            * "auto": automatically choose best initialization
            * "standard": initialize intercept with observed mean
            * "closed_form": try to initialize with closed form
        - np.ndarray: direct initialization of 'a'
    :param init_b: (Optional) Low-level initial values for b
        Can be:

        - str:
            * "auto": automatically choose best initialization
            * "standard": initialize with zeros
            * "closed_form": try to initialize with closed form
        - np.ndarray: direct initialization of 'b'
    :param gene_names: optional list/array of gene names which will be used if `data` does not implicitly store these
    :param sample_description: optional pandas.DataFrame containing sample annotations
    :param dmat_loc: Pre-built location model design matrix.
        This over-rides formula_loc and sample description information given in
        data or sample_description.
    :param dmat_scale: Pre-built scale model design matrix.
        This over-rides formula_scale and sample description information given in
        data or sample_description.
    :param constraints_loc: Constraints for location model. Can be one of the following:

            - np.ndarray:
                Array with constraints in rows and model parameters in columns.
                Each constraint contains non-zero entries for the a of parameters that
                has to sum to zero. This constraint is enforced by binding one parameter
                to the negative sum of the other parameters, effectively representing that
                parameter as a function of the other parameters. This dependent
                parameter is indicated by a -1 in this array, the independent parameters
                of that constraint (which may be dependent at an earlier constraint)
                are indicated by a 1. You should only use this option
                together with prebuilt design matrix for the location model, dmat_loc,
                for example via de.utils.setup_constrained().
            - dict:
                Every element of the dictionary corresponds to one set of equality constraints.
                Each set has to be be an entry of the form {..., x: y, ...}
                where x is the factor to be constrained and y is a factor by which levels of x are grouped
                and then constrained. Set y="1" to constrain all levels of x to sum to one,
                a single equality constraint.

                    E.g.: {"batch": "condition"} Batch levels within each condition are constrained to sum to
                        zero. This is applicable if repeats of a an experiment within each condition
                        are independent so that the set-up ~1+condition+batch is perfectly confounded.

                Can only group by non-constrained effects right now, use constraint_matrix_from_string
                for other cases.
            - list of strings or tuple of strings:
                String encoded equality constraints.

                    E.g. ["batch1 + batch2 + batch3 = 0"]
            - None:
                No constraints are used, this is equivalent to using an identity matrix as a
                constraint matrix.
    :param constraints_scale: Constraints for scale model. Can be one of the following:

            - np.ndarray:
                Array with constraints in rows and model parameters in columns.
                Each constraint contains non-zero entries for the a of parameters that
                has to sum to zero. This constraint is enforced by binding one parameter
                to the negative sum of the other parameters, effectively representing that
                parameter as a function of the other parameters. This dependent
                parameter is indicated by a -1 in this array, the independent parameters
                of that constraint (which may be dependent at an earlier constraint)
                are indicated by a 1. You should only use this option
                together with prebuilt design matrix for the scale model, dmat_scale,
                for example via de.utils.setup_constrained().
            - dict:
                Every element of the dictionary corresponds to one set of equality constraints.
                Each set has to be be an entry of the form {..., x: y, ...}
                where x is the factor to be constrained and y is a factor by which levels of x are grouped
                and then constrained. Set y="1" to constrain all levels of x to sum to one,
                a single equality constraint.

                    E.g.: {"batch": "condition"} Batch levels within each condition are constrained to sum to
                        zero. This is applicable if repeats of a an experiment within each condition
                        are independent so that the set-up ~1+condition+batch is perfectly confounded.

                Can only group by non-constrained effects right now, use constraint_matrix_from_string
                for other cases.
            - list of strings or tuple of strings:
                String encoded equality constraints.

                    E.g. ["batch1 + batch2 + batch3 = 0"]
            - None:
                No constraints are used, this is equivalent to using an identity matrix as a
                constraint matrix.
    :param size_factors: 1D array of transformed library size factors for each cell in the
        same order as in data or string-type column identifier of size-factor containing
        column in sample description.
    :param noise_model: str, noise model to use in model-based unit_test. Possible options:

        - 'nb': default
    :param batch_size: The batch size to use for the estimator.
    :param training_strategy: {str, function, list} training strategy to use. Can be:

        - str: will use Estimator.TrainingStrategy[training_strategy] to train
        - function: Can be used to implement custom training function will be called as
          `training_strategy(estimator)`.
        - list of keyword dicts containing method arguments: Will call Estimator.train() once with each dict of
          method arguments.
    :param quick_scale: Depending on the optimizer, `scale` will be fitted faster and maybe less accurate.

        Useful in scenarios where fitting the exact `scale` is not absolutely necessary.
    :param dtype: Allows specifying the precision which should be used to fit data.

        Should be "float32" for single precision or "float64" for double precision.
    :param kwargs: [Debugging] Additional arguments will be passed to the _fit method.
    :return:
        An estimator instance that contains all estimation relevant attributes and the model in estim.model.
        The attributes of the model depend on the noise model and the covariates used.
        We provide documentation for the model class in the model section of the documentation.
    """
    if len(kwargs) != 0:
        logging.getLogger("diffxpy").debug("additional kwargs: %s", str(kwargs))

    if (dmat_loc is None and formula_loc is None) or \
            (dmat_loc is not None and formula_loc is not None):
        raise ValueError("Supply either dmat_loc or formula_loc.")
    if (dmat_scale is None and formula_scale is None) or \
            (dmat_scale is not None and formula_scale != "~1"):
        raise ValueError("Supply either dmat_scale or formula_scale.")

    # # Parse input data formats:
    gene_names = parse_gene_names(data, gene_names)
    if dmat_loc is None and dmat_scale is None:
        sample_description = parse_sample_description(data, sample_description)
    size_factors = parse_size_factors(
        size_factors=size_factors,
        data=data,
        sample_description=sample_description
    )

    design_loc, constraints_loc = constraint_system_from_star(
        dmat=dmat_loc,
        sample_description=sample_description,
        formula=formula_loc,
        as_numeric=as_numeric,
        constraints=constraints_loc,
        return_type="patsy"
    )
    design_scale, constraints_scale = constraint_system_from_star(
        dmat=dmat_scale,
        sample_description=sample_description,
        formula=formula_scale,
        as_numeric=as_numeric,
        constraints=constraints_scale,
        return_type="patsy"
    )

    model = _fit(
        noise_model=noise_model,
        data=data,
        design_loc=design_loc,
        design_scale=design_scale,
        constraints_loc=constraints_loc,
        constraints_scale=constraints_scale,
        init_a=init_a,
        init_b=init_b,
        gene_names=gene_names,
        size_factors=size_factors,
        batch_size=batch_size,
        training_strategy=training_strategy,
        quick_scale=quick_scale,
        dtype=dtype
    )
    return model


def residuals(
        data: Union[anndata.AnnData, Raw, np.ndarray, scipy.sparse.csr_matrix, glm.typing.InputDataBase],
        formula_loc: Union[None, str] = None,
        formula_scale: Union[None, str] = "~1",
        as_numeric: Union[List[str], Tuple[str], str] = (),
        init_a: Union[np.ndarray, str] = "AUTO",
        init_b: Union[np.ndarray, str] = "AUTO",
        gene_names: Union[np.ndarray, list] = None,
        sample_description: Union[None, pd.DataFrame] = None,
        dmat_loc: Union[patsy.design_info.DesignMatrix] = None,
        dmat_scale: Union[patsy.design_info.DesignMatrix] = None,
        constraints_loc: Union[None, List[str], Tuple[str, str], dict, np.ndarray] = None,
        constraints_scale: Union[None, List[str], Tuple[str, str], dict, np.ndarray] = None,
        noise_model: str = "nb",
        size_factors: Union[np.ndarray, pd.core.series.Series, str] = None,
        batch_size: int = None,
        training_strategy: Union[str, List[Dict[str, object]], Callable] = "AUTO",
        quick_scale: bool = False,
        dtype="float64",
        **kwargs
):
    """
    Fits model for each gene and returns residuals.

    :param data: Input data matrix (observations x features) or (cells x genes).
    :param formula_loc: formula
        model formula for location and scale parameter models.
        If not specified, `formula` will be used instead.
    :param formula_scale: formula
        model formula for scale parameter model.
        If not specified, `formula` will be used instead.
    :param as_numeric:
        Which columns of sample_description to treat as numeric and
        not as categorical. This yields columns in the design matrix
        which do not correspond to one-hot encoded discrete factors.
        This makes sense for number of genes, time, pseudotime or space
        for example.
    :param init_a: (Optional) Low-level initial values for a.
        Can be:

        - str:
            * "auto": automatically choose best initialization
            * "standard": initialize intercept with observed mean
            * "closed_form": try to initialize with closed form
        - np.ndarray: direct initialization of 'a'
    :param init_b: (Optional) Low-level initial values for b
        Can be:

        - str:
            * "auto": automatically choose best initialization
            * "standard": initialize with zeros
            * "closed_form": try to initialize with closed form
        - np.ndarray: direct initialization of 'b'
    :param gene_names: optional list/array of gene names which will be used if `data` does not implicitly store these
    :param sample_description: optional pandas.DataFrame containing sample annotations
    :param dmat_loc: Pre-built location model design matrix.
        This over-rides formula_loc and sample description information given in
        data or sample_description.
    :param dmat_scale: Pre-built scale model design matrix.
        This over-rides formula_scale and sample description information given in
        data or sample_description.
    :param constraints_loc: Constraints for location model. Can be one of the following:

            - np.ndarray:
                Array with constraints in rows and model parameters in columns.
                Each constraint contains non-zero entries for the a of parameters that
                has to sum to zero. This constraint is enforced by binding one parameter
                to the negative sum of the other parameters, effectively representing that
                parameter as a function of the other parameters. This dependent
                parameter is indicated by a -1 in this array, the independent parameters
                of that constraint (which may be dependent at an earlier constraint)
                are indicated by a 1. You should only use this option
                together with prebuilt design matrix for the location model, dmat_loc,
                for example via de.utils.setup_constrained().
            - dict:
                Every element of the dictionary corresponds to one set of equality constraints.
                Each set has to be be an entry of the form {..., x: y, ...}
                where x is the factor to be constrained and y is a factor by which levels of x are grouped
                and then constrained. Set y="1" to constrain all levels of x to sum to one,
                a single equality constraint.

                    E.g.: {"batch": "condition"} Batch levels within each condition are constrained to sum to
                        zero. This is applicable if repeats of a an experiment within each condition
                        are independent so that the set-up ~1+condition+batch is perfectly confounded.

                Can only group by non-constrained effects right now, use constraint_matrix_from_string
                for other cases.
            - list of strings or tuple of strings:
                String encoded equality constraints.

                    E.g. ["batch1 + batch2 + batch3 = 0"]
            - None:
                No constraints are used, this is equivalent to using an identity matrix as a
                constraint matrix.
    :param constraints_scale: Constraints for scale model. Can be one of the following:

            - np.ndarray:
                Array with constraints in rows and model parameters in columns.
                Each constraint contains non-zero entries for the a of parameters that
                has to sum to zero. This constraint is enforced by binding one parameter
                to the negative sum of the other parameters, effectively representing that
                parameter as a function of the other parameters. This dependent
                parameter is indicated by a -1 in this array, the independent parameters
                of that constraint (which may be dependent at an earlier constraint)
                are indicated by a 1. You should only use this option
                together with prebuilt design matrix for the scale model, dmat_scale,
                for example via de.utils.setup_constrained().
            - dict:
                Every element of the dictionary corresponds to one set of equality constraints.
                Each set has to be be an entry of the form {..., x: y, ...}
                where x is the factor to be constrained and y is a factor by which levels of x are grouped
                and then constrained. Set y="1" to constrain all levels of x to sum to one,
                a single equality constraint.

                    E.g.: {"batch": "condition"} Batch levels within each condition are constrained to sum to
                        zero. This is applicable if repeats of a an experiment within each condition
                        are independent so that the set-up ~1+condition+batch is perfectly confounded.

                Can only group by non-constrained effects right now, use constraint_matrix_from_string
                for other cases.
            - list of strings or tuple of strings:
                String encoded equality constraints.

                    E.g. ["batch1 + batch2 + batch3 = 0"]
            - None:
                No constraints are used, this is equivalent to using an identity matrix as a
                constraint matrix.
    :param size_factors: 1D array of transformed library size factors for each cell in the
        same order as in data or string-type column identifier of size-factor containing
        column in sample description.
    :param noise_model: str, noise model to use in model-based unit_test. Possible options:

        - 'nb': default
    :param batch_size: The batch size to use for the estimator.
    :param training_strategy: {str, function, list} training strategy to use. Can be:

        - str: will use Estimator.TrainingStrategy[training_strategy] to train
        - function: Can be used to implement custom training function will be called as
          `training_strategy(estimator)`.
        - list of keyword dicts containing method arguments: Will call Estimator.train() once with each dict of
          method arguments.
    :param quick_scale: Depending on the optimizer, `scale` will be fitted faster and maybe less accurate.

        Useful in scenarios where fitting the exact `scale` is not absolutely necessary.
    :param dtype: Allows specifying the precision which should be used to fit data.

        Should be "float32" for single precision or "float64" for double precision.
    :param kwargs: [Debugging] Additional arguments will be passed to the _fit method.
    """
    estim = model(
        data=data,
        formula_loc=formula_loc,
        formula_scale=formula_scale,
        as_numeric=as_numeric,
        init_a=init_a,
        init_b=init_b,
        gene_names=gene_names,
        sample_description=sample_description,
        dmat_loc=dmat_loc,
        dmat_scale=dmat_scale,
        constraints_loc=constraints_loc,
        constraints_scale=constraints_scale,
        noise_model=noise_model,
        size_factors=size_factors,
        batch_size=batch_size,
        training_strategy=training_strategy,
        quick_scale=quick_scale,
        dtype=dtype,
        ** kwargs
    )
    residuals = estim.x - estim.model.location
    return residuals


def partition(
        data: Union[anndata.AnnData, Raw, np.ndarray, scipy.sparse.csr_matrix, glm.typing.InputDataBase],
        parts: Union[str, np.ndarray, list],
        gene_names: Union[np.ndarray, list] = None,
        sample_description: pd.DataFrame = None,
        dmat_loc: Union[patsy.design_info.DesignMatrix] = None,
        dmat_scale: Union[patsy.design_info.DesignMatrix] = None,
        size_factors: Union[np.ndarray, pd.core.series.Series, str] = None
):
    r"""
    Perform differential expression test for each group. This class handles
    the partitioning of the data set, the differential test callls and
    the sumamry of the individual tests into one
    DifferentialExpressionTestMulti object. All functions the yield
    DifferentialExpressionTestSingle objects can be performed on each
    partition.

    Wraps _Partition so that doc strings are nice.

    :param data: Array-like or anndata.Anndata object containing observations.
        Input data matrix (observations x features) or (cells x genes).
    :param parts: str, array

            - column in data.obs/sample_description which contains the split of observations into the two groups.
            - array of length `num_observations` containing group labels
    :param gene_names: optional list/array of gene names which will be used if `data` does not implicitly store these
    :param sample_description: optional pandas.DataFrame containing sample annotations.
    :param dmat_loc: Pre-built location model design matrix.
        This over-rides formula_loc and sample description information given in
        data or sample_description.
    :param dmat_scale: Pre-built scale model design matrix.
        This over-rides formula_scale and sample description information given in
        data or sample_description.
    :param size_factors: 1D array of transformed library size factors for each cell in the
        same order as in data or string-type column identifier of size-factor containing
        column in sample description.
    """
    return (_Partition(
        data=data,
        parts=parts,
        gene_names=gene_names,
        sample_description=sample_description,
        dmat_loc=dmat_loc,
        dmat_scale=dmat_scale,
        size_factors=size_factors
    ))


class _Partition:
    """
    Perform model fit separately for each group.
    """

    def __init__(
            self,
            data: Union[anndata.AnnData, Raw, np.ndarray, scipy.sparse.csr_matrix, glm.typing.InputDataBase],
            parts: Union[str, np.ndarray, list],
            gene_names: Union[np.ndarray, list] = None,
            sample_description: pd.DataFrame = None,
            dmat_loc: Union[patsy.design_info.DesignMatrix] = None,
            dmat_scale: Union[patsy.design_info.DesignMatrix] = None,
            size_factors: Union[np.ndarray, pd.core.series.Series, str] = None
    ):
        """
        :param data: Array-like or anndata.Anndata object containing observations.
            Input data matrix (observations x features) or (cells x genes).
        :param parts: str, array

            - column in data.obs/sample_description which contains the split of observations into the two groups.
            - array of length `num_observations` containing group labels
        :param gene_names: optional list/array of gene names which will be used if `data` does not implicitly store these
        :param sample_description: optional pandas.DataFrame containing sample annotations.
        :param dmat_loc: Pre-built location model design matrix.
            This over-rides formula_loc and sample description information given in
            data or sample_description.
        :param dmat_scale: Pre-built scale model design matrix.
            This over-rides formula_scale and sample description information given in
            data or sample_description.
        :param size_factors: 1D array of transformed library size factors for each cell in the
            same order as in data or string-type column identifier of size-factor containing
            column in sample description.
        """
        if isinstance(data, glm.typing.InputDataBase):
            self.x = data.x
        elif isinstance(data, anndata.AnnData) or isinstance(data, Raw):
            self.x = data.X
        elif isinstance(data, np.ndarray):
            self.x = data
        else:
            raise ValueError("data type %s not recognized" % type(data))
        self.gene_names = parse_gene_names(data, gene_names)
        self.sample_description = parse_sample_description(data, sample_description)
        self.dmat_loc = dmat_loc
        self.dmat_scale = dmat_scale
        self.size_factors = size_factors
        self.partition = parse_grouping(data, sample_description, parts)
        self.partitions = np.unique(self.partition)
        self.partition_idx = [np.where(self.partition == x)[0] for x in self.partitions]

    def model(
            self,
            formula_loc: Union[None, str] = None,
            formula_scale: Union[None, str] = "~1",
            as_numeric: Union[List[str], Tuple[str], str] = (),
            init_a: Union[np.ndarray, str] = "AUTO",
            init_b: Union[np.ndarray, str] = "AUTO",
            constraints_loc: Union[None, List[str], Tuple[str, str], dict, np.ndarray] = None,
            constraints_scale: Union[None, List[str], Tuple[str, str], dict, np.ndarray] = None,
            noise_model: str = "nb",
            batch_size: int = None,
            training_strategy: Union[str, List[Dict[str, object]], Callable] = "AUTO",
            quick_scale: bool = False,
            dtype="float64",
            **kwargs
    ):
        """
        Fit model for each gene and for each group.

        :param formula_loc: formula
            model formula for location and scale parameter models.
            If not specified, `formula` will be used instead.
        :param formula_scale: formula
            model formula for scale parameter model.
            If not specified, `formula` will be used instead.
        :param as_numeric:
            Which columns of sample_description to treat as numeric and
            not as categorical. This yields columns in the design matrix
            which do not correspond to one-hot encoded discrete factors.
            This makes sense for number of genes, time, pseudotime or space
            for example.
        :param init_a: (Optional) Low-level initial values for a.
            Can be:

            - str:
                * "auto": automatically choose best initialization
                * "standard": initialize intercept with observed mean
                * "closed_form": try to initialize with closed form
            - np.ndarray: direct initialization of 'a'
        :param init_b: (Optional) Low-level initial values for b
            Can be:

            - str:
                * "auto": automatically choose best initialization
                * "standard": initialize with zeros
                * "closed_form": try to initialize with closed form
            - np.ndarray: direct initialization of 'b'
        :param constraints_loc: Constraints for location model. Can be one of the following:

                - np.ndarray:
                    Array with constraints in rows and model parameters in columns.
                    Each constraint contains non-zero entries for the a of parameters that
                    has to sum to zero. This constraint is enforced by binding one parameter
                    to the negative sum of the other parameters, effectively representing that
                    parameter as a function of the other parameters. This dependent
                    parameter is indicated by a -1 in this array, the independent parameters
                    of that constraint (which may be dependent at an earlier constraint)
                    are indicated by a 1. You should only use this option
                    together with prebuilt design matrix for the location model, dmat_loc,
                    for example via de.utils.setup_constrained().
                - dict:
                    Every element of the dictionary corresponds to one set of equality constraints.
                    Each set has to be be an entry of the form {..., x: y, ...}
                    where x is the factor to be constrained and y is a factor by which levels of x are grouped
                    and then constrained. Set y="1" to constrain all levels of x to sum to one,
                    a single equality constraint.

                        E.g.: {"batch": "condition"} Batch levels within each condition are constrained to sum to
                            zero. This is applicable if repeats of a an experiment within each condition
                            are independent so that the set-up ~1+condition+batch is perfectly confounded.

                    Can only group by non-constrained effects right now, use constraint_matrix_from_string
                    for other cases.
                - list of strings or tuple of strings:
                    String encoded equality constraints.

                        E.g. ["batch1 + batch2 + batch3 = 0"]
                - None:
                    No constraints are used, this is equivalent to using an identity matrix as a
                    constraint matrix.
        :param constraints_scale: Constraints for scale model. Can be one of the following:

                - np.ndarray:
                    Array with constraints in rows and model parameters in columns.
                    Each constraint contains non-zero entries for the a of parameters that
                    has to sum to zero. This constraint is enforced by binding one parameter
                    to the negative sum of the other parameters, effectively representing that
                    parameter as a function of the other parameters. This dependent
                    parameter is indicated by a -1 in this array, the independent parameters
                    of that constraint (which may be dependent at an earlier constraint)
                    are indicated by a 1. You should only use this option
                    together with prebuilt design matrix for the scale model, dmat_scale,
                    for example via de.utils.setup_constrained().
                - dict:
                    Every element of the dictionary corresponds to one set of equality constraints.
                    Each set has to be be an entry of the form {..., x: y, ...}
                    where x is the factor to be constrained and y is a factor by which levels of x are grouped
                    and then constrained. Set y="1" to constrain all levels of x to sum to one,
                    a single equality constraint.

                        E.g.: {"batch": "condition"} Batch levels within each condition are constrained to sum to
                            zero. This is applicable if repeats of a an experiment within each condition
                            are independent so that the set-up ~1+condition+batch is perfectly confounded.

                    Can only group by non-constrained effects right now, use constraint_matrix_from_string
                    for other cases.
                - list of strings or tuple of strings:
                    String encoded equality constraints.

                        E.g. ["batch1 + batch2 + batch3 = 0"]
                - None:
                    No constraints are used, this is equivalent to using an identity matrix as a
                    constraint matrix.
        :param noise_model: str, noise model to use in model-based unit_test. Possible options:

            - 'nb': default
        :param batch_size: The batch size to use for the estimator.
        :param training_strategy: {str, function, list} training strategy to use. Can be:

            - str: will use Estimator.TrainingStrategy[training_strategy] to train
            - function: Can be used to implement custom training function will be called as
              `training_strategy(estimator)`.
            - list of keyword dicts containing method arguments: Will call Estimator.train() once with each dict of
              method arguments.
        :param quick_scale: Depending on the optimizer, `scale` will be fitted faster and maybe less accurate.

            Useful in scenarios where fitting the exact `scale` is not absolutely necessary.
        :param dtype: Allows specifying the precision which should be used to fit data.

            Should be "float32" for single precision or "float64" for double precision.
        :param kwargs: [Debugging] Additional arguments will be passed to the _fit method.
        """
        estims = []
        for i, idx in enumerate(self.partition_idx):
            estims.append(model(
                data=self.x[idx, :],
                formula_loc=formula_loc,
                formula_scale=formula_scale,
                as_numeric=as_numeric,
                init_a=init_a,
                init_b=init_b,
                gene_names=self.gene_names,
                sample_description=self.sample_description.iloc[idx, :],
                dmat_loc=self.dmat_loc[idx, :] if self.dmat_loc is not None else None,
                dmat_scale=self.dmat_scale[idx, :] if self.dmat_scale is not None else None,
                constraints_loc=constraints_loc,
                constraints_scale=constraints_scale,
                noise_model=noise_model,
                size_factors=self.size_factors[idx] if self.size_factors is not None else None,
                batch_size=batch_size,
                training_strategy=training_strategy,
                quick_scale=quick_scale,
                dtype=dtype,
                **kwargs
            ))
        return estims

    def residuals(
            self,
            formula_loc: Union[None, str] = None,
            formula_scale: Union[None, str] = "~1",
            as_numeric: Union[List[str], Tuple[str], str] = (),
            init_a: Union[np.ndarray, str] = "AUTO",
            init_b: Union[np.ndarray, str] = "AUTO",
            constraints_loc: Union[None, List[str], Tuple[str, str], dict, np.ndarray] = None,
            constraints_scale: Union[None, List[str], Tuple[str, str], dict, np.ndarray] = None,
            noise_model: str = "nb",
            batch_size: int = None,
            training_strategy: Union[str, List[Dict[str, object]], Callable] = "AUTO",
            quick_scale: bool = False,
            dtype="float64",
            **kwargs
    ):
        """
        Fit model for each gene and for each group and computes residuals.

        :param formula_loc: formula
            model formula for location and scale parameter models.
            If not specified, `formula` will be used instead.
        :param formula_scale: formula
            model formula for scale parameter model.
            If not specified, `formula` will be used instead.
        :param as_numeric:
            Which columns of sample_description to treat as numeric and
            not as categorical. This yields columns in the design matrix
            which do not correspond to one-hot encoded discrete factors.
            This makes sense for number of genes, time, pseudotime or space
            for example.
        :param init_a: (Optional) Low-level initial values for a.
            Can be:

            - str:
                * "auto": automatically choose best initialization
                * "standard": initialize intercept with observed mean
                * "closed_form": try to initialize with closed form
            - np.ndarray: direct initialization of 'a'
        :param init_b: (Optional) Low-level initial values for b
            Can be:

            - str:
                * "auto": automatically choose best initialization
                * "standard": initialize with zeros
                * "closed_form": try to initialize with closed form
            - np.ndarray: direct initialization of 'b'
        :param constraints_loc: Constraints for location model. Can be one of the following:

                - np.ndarray:
                    Array with constraints in rows and model parameters in columns.
                    Each constraint contains non-zero entries for the a of parameters that
                    has to sum to zero. This constraint is enforced by binding one parameter
                    to the negative sum of the other parameters, effectively representing that
                    parameter as a function of the other parameters. This dependent
                    parameter is indicated by a -1 in this array, the independent parameters
                    of that constraint (which may be dependent at an earlier constraint)
                    are indicated by a 1. You should only use this option
                    together with prebuilt design matrix for the location model, dmat_loc,
                    for example via de.utils.setup_constrained().
                - dict:
                    Every element of the dictionary corresponds to one set of equality constraints.
                    Each set has to be be an entry of the form {..., x: y, ...}
                    where x is the factor to be constrained and y is a factor by which levels of x are grouped
                    and then constrained. Set y="1" to constrain all levels of x to sum to one,
                    a single equality constraint.

                        E.g.: {"batch": "condition"} Batch levels within each condition are constrained to sum to
                            zero. This is applicable if repeats of a an experiment within each condition
                            are independent so that the set-up ~1+condition+batch is perfectly confounded.

                    Can only group by non-constrained effects right now, use constraint_matrix_from_string
                    for other cases.
                - list of strings or tuple of strings:
                    String encoded equality constraints.

                        E.g. ["batch1 + batch2 + batch3 = 0"]
                - None:
                    No constraints are used, this is equivalent to using an identity matrix as a
                    constraint matrix.
        :param constraints_scale: Constraints for scale model. Can be one of the following:

                - np.ndarray:
                    Array with constraints in rows and model parameters in columns.
                    Each constraint contains non-zero entries for the a of parameters that
                    has to sum to zero. This constraint is enforced by binding one parameter
                    to the negative sum of the other parameters, effectively representing that
                    parameter as a function of the other parameters. This dependent
                    parameter is indicated by a -1 in this array, the independent parameters
                    of that constraint (which may be dependent at an earlier constraint)
                    are indicated by a 1. You should only use this option
                    together with prebuilt design matrix for the scale model, dmat_scale,
                    for example via de.utils.setup_constrained().
                - dict:
                    Every element of the dictionary corresponds to one set of equality constraints.
                    Each set has to be be an entry of the form {..., x: y, ...}
                    where x is the factor to be constrained and y is a factor by which levels of x are grouped
                    and then constrained. Set y="1" to constrain all levels of x to sum to one,
                    a single equality constraint.

                        E.g.: {"batch": "condition"} Batch levels within each condition are constrained to sum to
                            zero. This is applicable if repeats of a an experiment within each condition
                            are independent so that the set-up ~1+condition+batch is perfectly confounded.

                    Can only group by non-constrained effects right now, use constraint_matrix_from_string
                    for other cases.
                - list of strings or tuple of strings:
                    String encoded equality constraints.

                        E.g. ["batch1 + batch2 + batch3 = 0"]
                - None:
                    No constraints are used, this is equivalent to using an identity matrix as a
                    constraint matrix.
        :param noise_model: str, noise model to use in model-based unit_test. Possible options:

            - 'nb': default
        :param batch_size: The batch size to use for the estimator.
        :param training_strategy: {str, function, list} training strategy to use. Can be:

            - str: will use Estimator.TrainingStrategy[training_strategy] to train
            - function: Can be used to implement custom training function will be called as
              `training_strategy(estimator)`.
            - list of keyword dicts containing method arguments: Will call Estimator.train() once with each dict of
              method arguments.
        :param quick_scale: Depending on the optimizer, `scale` will be fitted faster and maybe less accurate.

            Useful in scenarios where fitting the exact `scale` is not absolutely necessary.
        :param dtype: Allows specifying the precision which should be used to fit data.

            Should be "float32" for single precision or "float64" for double precision.
        :param kwargs: [Debugging] Additional arguments will be passed to the _fit method.
        """
        residuals = []
        for i, idx in enumerate(self.partition_idx):
            residuals.append(residuals(
                data=self.x[idx, :],
                formula_loc=formula_loc,
                formula_scale=formula_scale,
                as_numeric=as_numeric,
                init_a=init_a,
                init_b=init_b,
                gene_names=self.gene_names,
                sample_description=self.sample_description.iloc[idx, :],
                dmat_loc=self.dmat_loc[idx, :] if self.dmat_loc is not None else None,
                dmat_scale=self.dmat_scale[idx, :] if self.dmat_scale is not None else None,
                constraints_loc=constraints_loc,
                constraints_scale=constraints_scale,
                noise_model=noise_model,
                size_factors=self.size_factors[idx] if self.size_factors is not None else None,
                batch_size=batch_size,
                training_strategy=training_strategy,
                quick_scale=quick_scale,
                dtype=dtype,
                **kwargs
            ))
        return residuals


