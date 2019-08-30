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

from diffxpy import pkg_constants
from diffxpy.models.batch_bfgs.optim import Estim_BFGS
from .det import DifferentialExpressionTestLRT, DifferentialExpressionTestWald, \
    DifferentialExpressionTestTT, DifferentialExpressionTestRank, _DifferentialExpressionTestSingle, \
    DifferentialExpressionTestZTestLazy, DifferentialExpressionTestZTest, DifferentialExpressionTestPairwise, \
    DifferentialExpressionTestVsRest, _DifferentialExpressionTestMulti, DifferentialExpressionTestByPartition
from .det_cont import DifferentialExpressionTestWaldCont, DifferentialExpressionTestLRTCont
from .utils import parse_gene_names, parse_sample_description, parse_size_factors, parse_grouping, \
    constraint_system_from_star, design_matrix, preview_coef_names


def _fit(
        noise_model,
        data,
        design_loc,
        design_scale,
        constraints_loc: np.ndarray = None,
        constraints_scale: np.ndarray = None,
        init_model=None,
        init_a: Union[np.ndarray, str] = "AUTO",
        init_b: Union[np.ndarray, str] = "AUTO",
        gene_names=None,
        size_factors=None,
        batch_size: int = None,
        training_strategy: Union[str, List[Dict[str, object]], Callable] = "AUTO",
        quick_scale: bool = None,
        close_session=True,
        dtype="float64"
) -> glm.typing.InputDataBase:
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
    :param init_model: (optional) If provided, this model will be used to initialize this Estimator.
    :param init_a: (Optional) Low-level initial values for a.
        Can be:

        - str:
            * "auto": automatically choose best initialization
            * "standard": initialize intercept with observed mean
            * "init_model": initialize with another model (see `ìnit_model` parameter)
            * "closed_form": try to initialize with closed form
        - np.ndarray: direct initialization of 'a'
    :param init_b: (Optional) Low-level initial values for b
        Can be:

        - str:
            * "auto": automatically choose best initialization
            * "standard": initialize with zeros
            * "init_model": initialize with another model (see `ìnit_model` parameter)
            * "closed_form": try to initialize with closed form
        - np.ndarray: direct initialization of 'b'
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
    provide_optimizers = {
        "gd": pkg_constants.BATCHGLM_OPTIM_GD,
        "adam": pkg_constants.BATCHGLM_OPTIM_ADAM,
        "adagrad": pkg_constants.BATCHGLM_OPTIM_ADAGRAD,
        "rmsprop": pkg_constants.BATCHGLM_OPTIM_RMSPROP,
        "nr": pkg_constants.BATCHGLM_OPTIM_NEWTON,
        "nr_tr": pkg_constants.BATCHGLM_OPTIM_NEWTON_TR,
        "irls": pkg_constants.BATCHGLM_OPTIM_IRLS,
        "irls_gd": pkg_constants.BATCHGLM_OPTIM_IRLS_GD,
        "irls_tr": pkg_constants.BATCHGLM_OPTIM_IRLS_TR,
        "irls_gd_tr": pkg_constants.BATCHGLM_OPTIM_IRLS_GD_TR
    }

    if isinstance(training_strategy, str) and training_strategy.lower() == 'bfgs':
        assert False, "depreceated"
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
            from batchglm.api.models.glm_nb import Estimator, InputDataGLM
        elif noise_model == "norm" or noise_model == "normal":
            from batchglm.api.models.glm_norm import Estimator, InputDataGLM
        else:
            raise ValueError('base.test(): `noise_model="%s"` not recognized.' % noise_model)

        input_data = InputDataGLM(
            data=data,
            design_loc=design_loc,
            design_scale=design_scale,
            constraints_loc=constraints_loc,
            constraints_scale=constraints_scale,
            size_factors=size_factors,
            feature_names=gene_names,
        )

        constructor_args = {}
        if batch_size is not None:
            constructor_args["batch_size"] = batch_size
        if quick_scale is not None:
            constructor_args["quick_scale"] = quick_scale
        estim = Estimator(
            input_data=input_data,
            init_model=init_model,
            init_a=init_a,
            init_b=init_b,
            provide_optimizers=provide_optimizers,
            provide_batched=pkg_constants.BATCHGLM_PROVIDE_BATCHED,
            provide_fim=pkg_constants.BATCHGLM_PROVIDE_FIM,
            provide_hessian=pkg_constants.BATCHGLM_PROVIDE_HESSIAN,
            dtype=dtype,
            **constructor_args
        )
        estim.initialize()

        # Training:
        if callable(training_strategy):
            # call training_strategy if it is a function
            training_strategy(estim)
        else:
            estim.train_sequence(training_strategy=training_strategy)

        if close_session:
            estim.finalize()

    return estim


def lrt(
        data: Union[anndata.AnnData, Raw, np.ndarray, scipy.sparse.csr_matrix, glm.typing.InputDataBase],
        full_formula_loc: str,
        reduced_formula_loc: str,
        full_formula_scale: str = "~1",
        reduced_formula_scale: str = "~1",
        as_numeric: Union[List[str], Tuple[str], str] = (),
        init_a: Union[np.ndarray, str] = "AUTO",
        init_b: Union[np.ndarray, str] = "AUTO",
        gene_names: Union[np.ndarray, list] = None,
        sample_description: pd.DataFrame = None,
        noise_model="nb",
        size_factors: Union[np.ndarray, pd.core.series.Series, np.ndarray] = None,
        batch_size: int = None,
        training_strategy: Union[str, List[Dict[str, object]], Callable] = "AUTO",
        quick_scale: bool = False,
        dtype="float64",
        **kwargs
):
    """
    Perform log-likelihood ratio test for differential expression for each gene.

    Note that lrt() does not support constraints in its current form. Please
    use wald() for constraints.

    :param data: Input data matrix (observations x features) or (cells x genes).
    :param full_formula_loc: formula
        Full model formula for location parameter model.
        If not specified, `full_formula` will be used instead.
    :param reduced_formula_loc: formula
        Reduced model formula for location and scale parameter models.
        If not specified, `reduced_formula` will be used instead.
    :param full_formula_scale: formula
        Full model formula for scale parameter model.
        If not specified, `reduced_formula_scale` will be used instead.
    :param reduced_formula_scale: formula
        Reduced model formula for scale parameter model.
        If not specified, `reduced_formula` will be used instead.
    :param as_numeric:
        Which columns of sample_description to treat as numeric and
        not as categorical. This yields columns in the design matrix
        which do not correpond to one-hot encoded discrete factors.
        This makes sense for number of genes, time, pseudotime or space
        for example.
    :param init_a: (Optional) Low-level initial values for a.
        Can be:

        - str:
            * "auto": automatically choose best initialization
            * "standard": initialize intercept with observed mean
            * "init_model": initialize with another model (see `ìnit_model` parameter)
            * "closed_form": try to initialize with closed form
        - np.ndarray: direct initialization of 'a'
    :param init_b: (Optional) Low-level initial values for b
        Can be:

        - str:
            * "auto": automatically choose best initialization
            * "standard": initialize with zeros
            * "init_model": initialize with another model (see `ìnit_model` parameter)
            * "closed_form": try to initialize with closed form
        - np.ndarray: direct initialization of 'b'
    :param gene_names: optional list/array of gene names which will be used if `data` does not implicitly store these
    :param sample_description: optional pandas.DataFrame containing sample annotations
    :param noise_model: str, noise model to use in model-based unit_test. Possible options:

        - 'nb': default
    :param size_factors: 1D array of transformed library size factors for each cell in the
        same order as in data or string-type column identifier of size-factor containing
        column in sample description.
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
    # TODO test nestedness
    if len(kwargs) != 0:
        logging.getLogger("diffxpy").info("additional kwargs: %s", str(kwargs))

    if isinstance(as_numeric, str):
        as_numeric = [as_numeric]

    gene_names = parse_gene_names(data, gene_names)
    sample_description = parse_sample_description(data, sample_description)
    size_factors = parse_size_factors(
        size_factors=size_factors,
        data=data,
        sample_description=sample_description
    )

    full_design_loc = glm.data.design_matrix(
        sample_description=sample_description,
        formula=full_formula_loc,
        as_categorical=[False if x in as_numeric else True for x in sample_description.columns.values],
        return_type="patsy"
    )
    reduced_design_loc = glm.data.design_matrix(
        sample_description=sample_description,
        formula=reduced_formula_loc,
        as_categorical=[False if x in as_numeric else True for x in sample_description.columns.values],
        return_type="patsy"
    )
    full_design_scale = glm.data.design_matrix(
        sample_description=sample_description,
        formula=full_formula_scale,
        as_categorical=[False if x in as_numeric else True for x in sample_description.columns.values],
        return_type="patsy"
    )
    reduced_design_scale = glm.data.design_matrix(
        sample_description=sample_description,
        formula=reduced_formula_scale,
        as_categorical=[False if x in as_numeric else True for x in sample_description.columns.values],
        return_type="patsy"
    )

    reduced_model = _fit(
        noise_model=noise_model,
        data=data,
        design_loc=reduced_design_loc,
        design_scale=reduced_design_scale,
        constraints_loc=None,
        constraints_scale=None,
        init_a=init_a,
        init_b=init_b,
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
        data=data,
        design_loc=full_design_loc,
        design_scale=full_design_scale,
        constraints_loc=None,
        constraints_scale=None,
        gene_names=gene_names,
        init_a="init_model",
        init_b="init_model",
        init_model=reduced_model,
        size_factors=size_factors,
        batch_size=batch_size,
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
        data: Union[anndata.AnnData, Raw, np.ndarray, scipy.sparse.csr_matrix, glm.typing.InputDataBase],
        factor_loc_totest: Union[str, List[str]] = None,
        coef_to_test: Union[str, List[str]] = None,
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
    Perform Wald test for differential expression for each gene.

    :param data: Input data matrix (observations x features) or (cells x genes).
    :param factor_loc_totest: str, list of strings
        List of factors of formula to test with Wald test.
        E.g. "condition" or ["batch", "condition"] if formula_loc would be "~ 1 + batch + condition"
    :param coef_to_test:
        If there are more than two groups specified by `factor_loc_totest`,
        this parameter allows to specify the group which should be tested.
        Alternatively, if factor_loc_totest is not given, this list sets
        the exact coefficients which are to be tested.
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
    if len(kwargs) != 0:
        logging.getLogger("diffxpy").debug("additional kwargs: %s", str(kwargs))

    if (dmat_loc is None and formula_loc is None) or \
            (dmat_loc is not None and formula_loc is not None):
        raise ValueError("Supply either dmat_loc or formula_loc.")
    if (dmat_scale is None and formula_scale is None) or \
            (dmat_scale is not None and formula_scale != "~1"):
        raise ValueError("Supply either dmat_scale or formula_scale.")
    if dmat_loc is not None and factor_loc_totest is not None:
        raise ValueError("Supply coef_to_test and not factor_loc_totest if dmat_loc is supplied.")
    # Check that factor_loc_totest and coef_to_test are lists and not single strings:
    if isinstance(factor_loc_totest, str):
        factor_loc_totest = [factor_loc_totest]
    if isinstance(coef_to_test, str):
        coef_to_test = [coef_to_test]
    if isinstance(as_numeric, str):
        as_numeric = [as_numeric]

    # Parse input data formats:
    gene_names = parse_gene_names(data, gene_names)
    if dmat_loc is None and dmat_scale is None:
        sample_description = parse_sample_description(data, sample_description)
    size_factors = parse_size_factors(
        size_factors=size_factors,
        data=data,
        sample_description=sample_description
    )

    logging.getLogger("diffxpy").debug("building location model")
    design_loc, constraints_loc = constraint_system_from_star(
        dmat=dmat_loc,
        sample_description=sample_description,
        formula=formula_loc,
        as_numeric=as_numeric,
        constraints=constraints_loc,
        return_type="patsy"
    )
    logging.getLogger("diffxpy").debug("building scale model")
    design_scale, constraints_scale = constraint_system_from_star(
        dmat=dmat_scale,
        sample_description=sample_description,
        formula=formula_scale,
        as_numeric=as_numeric,
        constraints=constraints_scale,
        return_type="patsy"
    )

    # Define indices of coefficients to test:
    constraints_loc_temp = constraints_loc if constraints_loc is not None else np.eye(design_loc.shape[-1])
    if factor_loc_totest is not None:
        # Select coefficients to test via formula model:
        # Create temporary patsy design matrix to catch events in which design matrix is not patsy anymore here:
        design_loc_temp = design_matrix(
            data=data,
            sample_description=sample_description,
            formula=formula_loc,
            as_numeric=as_numeric,
            dmat=dmat_loc,
            return_type="patsy"
        )
        col_indices = np.concatenate([
            np.arange(design_loc_temp.shape[-1])[design_loc_temp.design_info.slice(x)]
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
        # Directly select coefficients to test from design matrix:
        if sample_description is not None:
            coef_loc_names = preview_coef_names(
                sample_description=sample_description,
                formula=formula_loc,
                as_numeric=as_numeric
            ).tolist()
        else:
            coef_loc_names = dmat_loc.columns.tolist()
        if not np.all([x in coef_loc_names for x in coef_to_test]):
            raise ValueError(
                "the requested test coefficients %s were found in model coefficients %s" %
                (", ".join([x for x in coef_to_test if x not in coef_loc_names]),
                 ", ".join(coef_loc_names))
            )
        col_indices = np.asarray([
            coef_loc_names.index(x) for x in coef_to_test
        ])
    else:
        raise ValueError("either set factor_loc_totest or coef_to_test")
    # Check that all tested coefficients are independent:
    for x in col_indices:
        if np.sum(constraints_loc_temp[x,:]) != 1:
            raise ValueError("Constraints input is wrong: not all tested coefficients are unconstrained.")
    # Adjust tested coefficients from dependent to independent (fitted) parameters:
    col_indices = np.array([np.where(constraints_loc_temp[x,:] == 1)[0][0] for x in col_indices])

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
        dtype=dtype,
        **kwargs,
    )

    de_test = DifferentialExpressionTestWald(
        model_estim=model,
        col_indices=col_indices,
        noise_model=noise_model,
        sample_description=sample_description
    )

    return de_test


def t_test(
        data: Union[anndata.AnnData, Raw, np.ndarray, scipy.sparse.csr_matrix, glm.typing.InputDataBase],
        grouping,
        gene_names: Union[np.ndarray, list] = None,
        sample_description: pd.DataFrame = None,
        is_logged: bool = False,
        is_sig_zerovar: bool = True
):
    """
    Perform Welch's t-test for differential expression
    between two groups on adata object for each gene.

    :param data: Array-like, or anndata.Anndata object containing observations.
        Input data matrix (observations x features) or (cells x genes).
    :param grouping: str, array

        - column in data.obs/sample_description which contains the split of observations into the two groups.
        - array of length `num_observations` containing group labels
    :param gene_names: optional list/array of gene names which will be used if `data` does not implicitly store these
    :param sample_description: optional pandas.DataFrame containing sample annotations
    :param is_logged:
        Whether data is already logged. If True, log-fold changes are computed as fold changes on this data.
        If False, log-fold changes are computed as log-fold changes on this data.
    :param is_sig_zerovar:
        Whether to assign p-value of 0 to a gene which has zero variance in both groups but not the same mean. If False,
        the p-value is set to np.nan.
    """
    gene_names = parse_gene_names(data, gene_names)
    grouping = parse_grouping(data, sample_description, grouping)

    de_test = DifferentialExpressionTestTT(
        data=data,
        sample_description=sample_description,
        grouping=grouping,
        gene_names=gene_names,
        is_logged=is_logged,
        is_sig_zerovar=is_sig_zerovar
    )

    return de_test


def rank_test(
        data: Union[anndata.AnnData, Raw, np.ndarray, scipy.sparse.csr_matrix, glm.typing.InputDataBase],
        grouping: Union[str, np.ndarray, list],
        gene_names: Union[np.ndarray, list] = None,
        sample_description: pd.DataFrame = None,
        is_logged: bool = False,
        is_sig_zerovar: bool = True
):
    """
    Perform Mann-Whitney rank test (Wilcoxon rank-sum test) for differential expression
    between two groups on adata object for each gene.

    :param data: Array-like, or anndata.Anndata object containing observations.
        Input data matrix (observations x features) or (cells x genes).
    :param grouping: str, array

        - column in data.obs/sample_description which contains the split of observations into the two groups.
        - array of length `num_observations` containing group labels
    :param gene_names: optional list/array of gene names which will be used if `data` does not implicitly store these
    :param sample_description: optional pandas.DataFrame containing sample annotations
    :param is_logged:
        Whether data is already logged. If True, log-fold changes are computed as fold changes on this data.
        If False, log-fold changes are computed as log-fold changes on this data.
    :param is_sig_zerovar:
        Whether to assign p-value of 0 to a gene which has zero variance in both groups but not the same mean. If False,
        the p-value is set to np.nan.
    """
    gene_names = parse_gene_names(data, gene_names)
    grouping = parse_grouping(data, sample_description, grouping)

    de_test = DifferentialExpressionTestRank(
        data=data,
        sample_description=sample_description,
        grouping=grouping,
        gene_names=gene_names,
        is_logged=is_logged,
        is_sig_zerovar=is_sig_zerovar
    )

    return de_test


def two_sample(
        data: Union[anndata.AnnData, Raw, np.ndarray, scipy.sparse.csr_matrix, glm.typing.InputDataBase],
        grouping: Union[str, np.ndarray, list],
        as_numeric: Union[List[str], Tuple[str], str] = (),
        test: str = "t-test",
        gene_names: Union[np.ndarray, list] = None,
        sample_description: pd.DataFrame = None,
        noise_model: str = None,
        size_factors: np.ndarray = None,
        batch_size: int = None,
        training_strategy: Union[str, List[Dict[str, object]], Callable] = "AUTO",
        is_sig_zerovar: bool = True,
        quick_scale: bool = None,
        dtype="float64",
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

    - "lrt" - (log-likelihood ratio test):
        Requires the fitting of 2 generalized linear models (full and reduced).
        The models are automatically assembled as follows, use the de.test.lrt()
        function if you would like to perform a different test.

            * full model location parameter: ~ 1 + group
            * full model scale parameter: ~ 1 + group
            * reduced model location parameter: ~ 1
            * reduced model scale parameter: ~ 1 + group
    - "wald" - Wald test:
        Requires the fitting of 1 generalized linear models.
        model location parameter: ~ 1 + group
        model scale parameter: ~ 1 + group
        Test the group coefficient of the location parameter model against 0.
    - "t-test" - Welch's t-test:
        Doesn't require fitting of generalized linear models.
        Welch's t-test between both observation groups.
    - "rank" - Wilcoxon rank sum (Mann-Whitney U) test:
        Doesn't require fitting of generalized linear models.
        Wilcoxon rank sum (Mann-Whitney U) test between both observation groups.

    :param data: Array-like, or anndata.Anndata object containing observations.
        Input data matrix (observations x features) or (cells x genes).
    :param grouping: str, array

        - column in data.obs/sample_description which contains the split of observations into the two groups.
        - array of length `num_observations` containing group labels
    :param as_numeric:
        Which columns of sample_description to treat as numeric and
        not as categorical. This yields columns in the design matrix
        which do not correpond to one-hot encoded discrete factors.
        This makes sense for number of genes, time, pseudotime or space
        for example.
    :param test: str, statistical test to use. Possible options:

        - 'wald': default
        - 'lrt'
        - 't-test'
        - 'rank'
    :param gene_names: optional list/array of gene names which will be used if `data` does not implicitly store these
    :param sample_description: optional pandas.DataFrame containing sample annotations
    :param size_factors: 1D array of transformed library size factors for each cell in the
        same order as in data
    :param noise_model: str, noise model to use in model-based unit_test. Possible options:

        - 'nb': default
    :param batch_size: The batch size to use for the estimator.
    :param training_strategy: {str, function, list} training strategy to use. Can be:

        - str: will use Estimator.TrainingStrategy[training_strategy] to train
        - function: Can be used to implement custom training function will be called as
          `training_strategy(estimator)`.
        - list of keyword dicts containing method arguments: Will call Estimator.train() once with each dict of
          method arguments.
    :param is_sig_zerovar:
        Whether to assign p-value of 0 to a gene which has zero variance in both groups but not the same mean. If False,
        the p-value is set to np.nan.
    :param quick_scale: Depending on the optimizer, `scale` will be fitted faster and maybe less accurate.

        Useful in scenarios where fitting the exact `scale` is not absolutely necessary.
    :param dtype: Allows specifying the precision which should be used to fit data.

        Should be "float32" for single precision or "float64" for double precision.
    :param kwargs: [Debugging] Additional arguments will be passed to the _fit method.
    """
    if test in ['t-test', 'rank'] and noise_model is not None:
        raise Warning('two_sample(): Do not specify `noise_model` if using test t-test or rank_test: ' +
                      'The t-test is based on a gaussian noise model and the rank sum test is model free.')

    gene_names = parse_gene_names(data, gene_names)
    grouping = parse_grouping(data, sample_description, grouping)
    sample_description = pd.DataFrame({"grouping": grouping})

    groups = np.unique(grouping)
    if groups.size > 2:
        raise ValueError("More than two groups detected:\n\t%s", groups)
    if groups.size < 2:
        raise ValueError("Less than two groups detected:\n\t%s", groups)

    if test.lower() == 'wald':
        if noise_model is None:
            raise ValueError("Please specify noise_model")
        formula_loc = '~ 1 + grouping'
        formula_scale = '~ 1 + grouping'
        de_test = wald(
            data=data,
            factor_loc_totest="grouping",
            as_numeric=as_numeric,
            coef_to_test=None,
            formula_loc=formula_loc,
            formula_scale=formula_scale,
            gene_names=gene_names,
            sample_description=sample_description,
            noise_model=noise_model,
            size_factors=size_factors,
            init_a="closed_form",
            init_b="closed_form",
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
        full_formula_scale = '~ 1'
        reduced_formula_loc = '~ 1'
        reduced_formula_scale = '~ 1'
        de_test = lrt(
            data=data,
            full_formula_loc=full_formula_loc,
            reduced_formula_loc=reduced_formula_loc,
            full_formula_scale=full_formula_scale,
            reduced_formula_scale=reduced_formula_scale,
            as_numeric=as_numeric,
            gene_names=gene_names,
            sample_description=sample_description,
            noise_model=noise_model,
            size_factors=size_factors,
            init_a="closed_form",
            init_b="closed_form",
            batch_size=batch_size,
            training_strategy=training_strategy,
            quick_scale=quick_scale,
            dtype=dtype,
            **kwargs
        )
    elif test.lower() == 't-test' or test.lower() == "t_test" or test.lower() == "ttest":
        de_test = t_test(
            data=data,
            gene_names=gene_names,
            grouping=grouping,
            is_sig_zerovar=is_sig_zerovar
        )
    elif test.lower() == 'rank':
        de_test = rank_test(
            data=data,
            gene_names=gene_names,
            grouping=grouping,
            is_sig_zerovar=is_sig_zerovar
        )
    else:
        raise ValueError('two_sample(): Parameter `test="%s"` not recognized.' % test)

    return de_test


def pairwise(
        data: Union[anndata.AnnData, Raw, np.ndarray, scipy.sparse.csr_matrix, glm.typing.InputDataBase],
        grouping: Union[str, np.ndarray, list],
        as_numeric: Union[List[str], Tuple[str], str] = (),
        test: str = "z-test",
        lazy: bool = True,
        gene_names: Union[np.ndarray, list] = None,
        sample_description: pd.DataFrame = None,
        noise_model: str = "nb",
        size_factors: np.ndarray = None,
        batch_size: int = None,
        training_strategy: Union[str, List[Dict[str, object]], Callable] = "AUTO",
        is_sig_zerovar: bool = True,
        quick_scale: bool = False,
        dtype="float64",
        pval_correction: str = "global",
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

    - "lrt" -log-likelihood ratio test:
        Requires the fitting of 2 generalized linear models (full and reduced).

        * full model location parameter: ~ 1 + group
        * full model scale parameter: ~ 1 + group
        * reduced model location parameter: ~ 1
        * reduced model scale parameter: ~ 1 + group
    - "wald" - Wald test:
        Requires the fitting of 1 generalized linear models.
        model location parameter: ~ 1 + group
        model scale parameter: ~ 1 + group
        Test the group coefficient of the location parameter model against 0.
    - "t-test" - Welch's t-test:
        Doesn't require fitting of generalized linear models.
        Welch's t-test between both observation groups.
    - "rank" - Wilcoxon rank sum (Mann-Whitney U) test:
        Doesn't require fitting of generalized linear models.
        Wilcoxon rank sum (Mann-Whitney U) test between both observation groups.

    :param data: Array-like, or anndata.Anndata object containing observations.
        Input data matrix (observations x features) or (cells x genes).
    :param grouping: str, array

        - column in data.obs/sample_description which contains the split of observations into the two groups.
        - array of length `num_observations` containing group labels
    :param as_numeric:
        Which columns of sample_description to treat as numeric and
        not as categorical. This yields columns in the design matrix
        which do not correpond to one-hot encoded discrete factors.
        This makes sense for number of genes, time, pseudotime or space
        for example.
    :param test: str, statistical test to use. Possible options:

        - 'z-test': default
        - 'wald'
        - 'lrt'
        - 't-test'
        - 'rank'
    :param lazy: bool, whether to enable lazy results evaluation.
        This is only possible if test=="ztest" and yields an output object which computes
        p-values etc. only upon request of certain pairs. This makes sense if the entire
        gene x groups x groups matrix which contains all pairwise p-values, q-values or
        log-fold changes is very large and may not fit into memory, especially if only
        a certain subset of the pairwise comparisons is desired anyway.
    :param gene_names: optional list/array of gene names which will be used if `data` does not implicitly store these
    :param sample_description: optional pandas.DataFrame containing sample annotations
    :param size_factors: 1D array of transformed library size factors for each cell in the
        same order as in data
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
    :param pval_correction: Choose between global and test-wise correction.
        Can be:

        - "global": correct all p-values in one operation
        - "by_test": correct the p-values of each test individually
    :param keep_full_test_objs: [Debugging] keep the individual test objects; currently valid for test != "z-test".
    :param is_sig_zerovar:
        Whether to assign p-value of 0 to a gene which has zero variance in both groups but not the same mean. If False,
        the p-value is set to np.nan.
    :param kwargs: [Debugging] Additional arguments will be passed to the _fit method.
    """
    if len(kwargs) != 0:
        logging.getLogger("diffxpy").info("additional kwargs: %s", str(kwargs))

    if lazy and not (test.lower() == 'z-test' or test.lower() == 'z_test' or test.lower() == 'ztest'):
        raise ValueError("lazy evaluation of pairwise tests only possible if test is z-test")

    # Do not store all models but only p-value and q-value matrix:
    # genes x groups x groups
    gene_names = parse_gene_names(data, gene_names)
    sample_description = parse_sample_description(data, sample_description)
    grouping = parse_grouping(data, sample_description, grouping)
    sample_description = pd.DataFrame({"grouping": grouping})

    if test.lower() == 'z-test' or test.lower() == 'z_test' or test.lower() == 'ztest':
        # -1 in formula removes intercept
        dmat = glm.data.design_matrix(
            sample_description,
            formula="~ 1 - 1 + grouping"
        )
        model = _fit(
            noise_model=noise_model,
            data=data,
            design_loc=dmat,
            design_scale=dmat,
            gene_names=gene_names,
            size_factors=size_factors,
            init_a="closed_form",
            init_b="closed_form",
            batch_size=batch_size,
            training_strategy=training_strategy,
            quick_scale=quick_scale,
            dtype=dtype,
            **kwargs
        )

        if lazy:
            de_test = DifferentialExpressionTestZTestLazy(
                model_estim=model,
                grouping=grouping,
                groups=np.unique(grouping),
                correction_type=pval_correction
            )
        else:
            de_test = DifferentialExpressionTestZTest(
                model_estim=model,
                grouping=grouping,
                groups=np.unique(grouping),
                correction_type=pval_correction
            )
    else:
        if isinstance(data, anndata.AnnData) or isinstance(data, anndata.Raw):
            data = data.X
        elif isinstance(data, glm.typing.InputDataBase):
            data = data.x
        groups = np.unique(grouping)
        pvals = np.tile(np.NaN, [len(groups), len(groups), data.shape[1]])
        pvals[np.eye(pvals.shape[0]).astype(bool)] = 0
        logfc = np.tile(np.NaN, [len(groups), len(groups), data.shape[1]])
        logfc[np.eye(logfc.shape[0]).astype(bool)] = 0

        if keep_full_test_objs:
            tests = np.tile([None], [len(groups), len(groups)])
        else:
            tests = None

        for i, g1 in enumerate(groups):
            for j, g2 in enumerate(groups[(i + 1):]):
                j = j + i + 1

                idx = np.where(np.logical_or(
                    grouping == g1,
                    grouping == g2
                ))[0]
                de_test_temp = two_sample(
                    data=data[idx, :],
                    grouping=grouping[idx],
                    as_numeric=as_numeric,
                    test=test,
                    gene_names=gene_names,
                    sample_description=sample_description.iloc[idx, :],
                    noise_model=noise_model,
                    size_factors=size_factors[idx] if size_factors is not None else None,
                    batch_size=batch_size,
                    training_strategy=training_strategy,
                    quick_scale=quick_scale,
                    is_sig_zerovar=is_sig_zerovar,
                    dtype=dtype,
                    **kwargs
                )
                pvals[i, j] = de_test_temp.pval
                pvals[j, i] = pvals[i, j]
                logfc[i, j] = de_test_temp.log_fold_change()
                logfc[j, i] = -logfc[i, j]
                if keep_full_test_objs:
                    tests[i, j] = de_test_temp
                    tests[j, i] = de_test_temp

        de_test = DifferentialExpressionTestPairwise(
            gene_ids=gene_names,
            pval=pvals,
            logfc=logfc,
            ave=np.mean(data, axis=0),
            groups=groups,
            tests=tests,
            correction_type=pval_correction
        )

    return de_test


def versus_rest(
        data: Union[anndata.AnnData, Raw, np.ndarray, scipy.sparse.csr_matrix, glm.typing.InputDataBase],
        grouping: Union[str, np.ndarray, list],
        as_numeric: Union[List[str], Tuple[str], str] = (),
        test: str = 'wald',
        gene_names: Union[np.ndarray, list] = None,
        sample_description: pd.DataFrame = None,
        noise_model: str = None,
        size_factors: np.ndarray = None,
        batch_size: int = None,
        training_strategy: Union[str, List[Dict[str, object]], Callable] = "AUTO",
        is_sig_zerovar: bool = True,
        quick_scale: bool = None,
        dtype="float64",
        pval_correction: str = "global",
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

    - "lrt" - log-likelihood ratio test):
        Requires the fitting of 2 generalized linear models (full and reduced).

        * full model location parameter: ~ 1 + group
        * full model scale parameter: ~ 1 + group
        * reduced model location parameter: ~ 1
        * reduced model scale parameter: ~ 1 + group
    - "wald" - Wald test:
        Requires the fitting of 1 generalized linear models.
        model location parameter: ~ 1 + group
        model scale parameter: ~ 1 + group
        Test the group coefficient of the location parameter model against 0.
    - "t-test" - Welch's t-test:
        Doesn't require fitting of generalized linear models.
        Welch's t-test between both observation groups.
    - "rank" - Wilcoxon rank sum (Mann-Whitney U) test:
        Doesn't require fitting of generalized linear models.
        Wilcoxon rank sum (Mann-Whitney U) test between both observation groups.

    :param data: Array-like or anndata.Anndata object containing observations.
        Input data matrix (observations x features) or (cells x genes).
    :param grouping: str, array

        - column in data.obs/sample_description which contains the split of observations into the two groups.
        - array of length `num_observations` containing group labels
    :param as_numeric:
        Which columns of sample_description to treat as numeric and
        not as categorical. This yields columns in the design matrix
        which do not correpond to one-hot encoded discrete factors.
        This makes sense for number of genes, time, pseudotime or space
        for example.
    :param test: str, statistical test to use. Possible options (see function description):

        - 'wald'
        - 'lrt'
        - 't-test'
        - 'rank'
    :param gene_names: optional list/array of gene names which will be used if `data` does not implicitly store these
    :param sample_description: optional pandas.DataFrame containing sample annotations
    :param pval_correction: Choose between global and test-wise correction.
        Can be:

        - "global": correct all p-values in one operation
        - "by_test": correct the p-values of each test individually
    :param size_factors: 1D array of transformed library size factors for each cell in the
        same order as in data
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
    :param pval_correction: Choose between global and test-wise correction.
        Can be:

        - "global": correct all p-values in one operation
        - "by_test": correct the p-values of each test individually
    :param is_sig_zerovar:
        Whether to assign p-value of 0 to a gene which has zero variance in both groups but not the same mean. If False,
        the p-value is set to np.nan.
    :param kwargs: [Debugging] Additional arguments will be passed to the _fit method.
    """
    if len(kwargs) != 0:
        logging.getLogger("diffxpy").info("additional kwargs: %s", str(kwargs))

    # Do not store all models but only p-value and q-value matrix:
    # genes x groups
    gene_names = parse_gene_names(data, gene_names)
    sample_description = parse_sample_description(data, sample_description)
    grouping = parse_grouping(data, sample_description, grouping)
    sample_description = pd.DataFrame({"grouping": grouping})

    groups = np.unique(grouping)
    pvals = np.zeros([1, len(groups), data.shape[1]])
    logfc = np.zeros([1, len(groups), data.shape[1]])

    if keep_full_test_objs:
        tests = np.tile([None], [1, len(groups)])
    else:
        tests = None

    for i, g1 in enumerate(groups):
        test_grouping = np.where(grouping == g1, "group", "rest")
        de_test_temp = two_sample(
            data=data,
            grouping=test_grouping,
            as_numeric=as_numeric,
            test=test,
            gene_names=gene_names,
            sample_description=sample_description,
            noise_model=noise_model,
            batch_size=batch_size,
            training_strategy=training_strategy,
            quick_scale=quick_scale,
            size_factors=size_factors,
            is_sig_zerovar=is_sig_zerovar,
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
        ave=np.mean(data, axis=0),
        groups=groups,
        tests=tests,
        correction_type=pval_correction
    )

    return de_test


def partition(
        data: Union[anndata.AnnData, Raw, np.ndarray, scipy.sparse.csr_matrix, glm.typing.InputDataBase],
        parts: Union[str, np.ndarray, list],
        gene_names: Union[np.ndarray, list] = None,
        sample_description: pd.DataFrame = None
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
    :param sample_description: optional pandas.DataFrame containing sample annotations
    """
    return (_Partition(
        data=data,
        parts=parts,
        gene_names=gene_names,
        sample_description=sample_description))


class _Partition:
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
            data: Union[anndata.AnnData, Raw, np.ndarray, scipy.sparse.csr_matrix, glm.typing.InputDataBase],
            parts: Union[str, np.ndarray, list],
            gene_names: Union[np.ndarray, list] = None,
            sample_description: pd.DataFrame = None
    ):
        """
        :param data: Array-like or anndata.Anndata object containing observations.
            Input data matrix (observations x features) or (cells x genes).
        :param parts: str, array

            - column in data.obs/sample_description which contains the split of observations into the two groups.
            - array of length `num_observations` containing group labels
        :param gene_names: optional list/array of gene names which will be used if `data` does not implicitly store these
        :param sample_description: optional pandas.DataFrame containing sample annotations
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
        self.partition = parse_grouping(data, sample_description, parts)
        self.partitions = np.unique(self.partition)
        self.partition_idx = [np.where(self.partition == x)[0] for x in self.partitions]

    def two_sample(
            self,
            grouping: Union[str],
            as_numeric: Union[List[str], Tuple[str], str] = (),
            test=None,
            size_factors: np.ndarray = None,
            noise_model: str = None,
            batch_size: int = None,
            training_strategy: Union[str, List[Dict[str, object]], Callable] = "AUTO",
            is_sig_zerovar: bool = True,
            **kwargs
    ) -> _DifferentialExpressionTestMulti:
        """
        Performs a two-sample test within each partition of a data set.

        See also annotation of de.test.two_sample()

        :param grouping: str

            - column in data.obs/sample_description which contains the split of observations into the two groups.
        :param as_numeric:
            Which columns of sample_description to treat as numeric and
            not as categorical. This yields columns in the design matrix
            which do not correpond to one-hot encoded discrete factors.
            This makes sense for number of genes, time, pseudotime or space
            for example.
        :param test: str, statistical test to use. Possible options:

            - 'wald': default
            - 'lrt'
            - 't-test'
            - 'rank'
        :param size_factors: 1D array of transformed library size factors for each cell in the
            same order as in data
        :param noise_model: str, noise model to use in model-based unit_test. Possible options:

            - 'nb': default
        :param batch_size: The batch size to use for the estimator.
        :param training_strategy: {str, function, list} training strategy to use. Can be:

            - str: will use Estimator.TrainingStrategy[training_strategy] to train
            - function: Can be used to implement custom training function will be called as
              `training_strategy(estimator)`.
            - list of keyword dicts containing method arguments: Will call Estimator.train() once with each dict of
              method arguments.
        :param is_sig_zerovar:
            Whether to assign p-value of 0 to a gene which has zero variance in both groups but not the same mean. If False,
            the p-value is set to np.nan.
        :param kwargs: [Debugging] Additional arguments will be passed to the _fit method.
        """
        DETestsSingle = []
        for i, idx in enumerate(self.partition_idx):
            DETestsSingle.append(two_sample(
                data=self.x[idx, :],
                grouping=grouping,
                as_numeric=as_numeric,
                test=test,
                gene_names=self.gene_names,
                sample_description=self.sample_description.iloc[idx, :],
                noise_model=noise_model,
                size_factors=size_factors[idx] if size_factors is not None else None,
                batch_size=batch_size,
                training_strategy=training_strategy,
                is_sig_zerovar=is_sig_zerovar,
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
            is_logged: bool,
            is_sig_zerovar: bool = True,
            dtype="float64"
    ):
        """
        Performs a Welch's t-test within each partition of a data set.

        See also annotation of de.test.t_test()

        :param grouping: str

            - column in data.obs/sample_description which contains the split of observations into the two groups.
        :param is_logged:
            Whether data is already logged. If True, log-fold changes are computed as fold changes on this data.
            If False, log-fold changes are computed as log-fold changes on this data.
        :param is_sig_zerovar:
            Whether to assign p-value of 0 to a gene which has zero variance in both groups but not the same mean. If False,
            the p-value is set to np.nan.
        :param dtype:
        """
        DETestsSingle = []
        for i, idx in enumerate(self.partition_idx):
            DETestsSingle.append(t_test(
                data=self.x[idx, :],
                grouping=grouping,
                is_logged=is_logged,
                gene_names=self.gene_names,
                sample_description=self.sample_description.iloc[idx, :],
                is_sig_zerovar=is_sig_zerovar,
                dtype=dtype
            ))
        return DifferentialExpressionTestByPartition(
            partitions=self.partitions,
            tests=DETestsSingle,
            ave=np.mean(self.x, axis=0),
            correction_type="by_test")

    def rank_test(
            self,
            grouping: Union[str],
            is_sig_zerovar: bool = True,
            dtype="float64"
    ):
        """
        Performs a Wilcoxon rank sum test within each partition of a data set.

        See also annotation of de.test.rank_test()

        :param grouping: str, array

            - column in data.obs/sample_description which contains the split of observations into the two groups.
            - array of length `num_observations` containing group labels
        :param is_sig_zerovar:
            Whether to assign p-value of 0 to a gene which has zero variance in both groups but not the same mean. If False,
            the p-value is set to np.nan.
        :param dtype:
        """
        DETestsSingle = []
        for i, idx in enumerate(self.partition_idx):
            DETestsSingle.append(rank_test(
                data=self.x[idx, :],
                grouping=grouping,
                gene_names=self.gene_names,
                sample_description=self.sample_description.iloc[idx, :],
                is_sig_zerovar=is_sig_zerovar,
                dtype=dtype
            ))
        return DifferentialExpressionTestByPartition(
            partitions=self.partitions,
            tests=DETestsSingle,
            ave=np.mean(self.x, axis=0),
            correction_type="by_test")

    def lrt(
            self,
            full_formula_loc: str = None,
            reduced_formula_loc: str = None,
            full_formula_scale: str = "~1",
            reduced_formula_scale: str = None,
            as_numeric: Union[List[str], Tuple[str], str] = (),
            init_a: Union[str] = "AUTO",
            init_b: Union[str] = "AUTO",
            size_factors: np.ndarray = None,
            noise_model="nb",
            batch_size: int = None,
            training_strategy: Union[str, List[Dict[str, object]], Callable] = "AUTO",
            **kwargs
    ):
        """
        Performs a likelihood-ratio test within each partition of a data set.

        See also annotation of de.test.lrt()

        :param full_formula_loc: formula
            Full model formula for location parameter model.
            If not specified, `full_formula` will be used instead.
        :param reduced_formula_loc: formula
            Reduced model formula for location and scale parameter models.
            If not specified, `reduced_formula` will be used instead.
        :param full_formula_scale: formula
            Full model formula for scale parameter model.
            If not specified, `reduced_formula_scale` will be used instead.
        :param reduced_formula_scale: formula
            Reduced model formula for scale parameter model.
            If not specified, `reduced_formula` will be used instead.
        :param as_numeric:
            Which columns of sample_description to treat as numeric and
            not as categorical. This yields columns in the design matrix
            which do not correpond to one-hot encoded discrete factors.
            This makes sense for number of genes, time, pseudotime or space
            for example.
        :param init_a: (Optional) Low-level initial values for a.
            Can be:

            - str:
                * "auto": automatically choose best initialization
                * "standard": initialize intercept with observed mean
                * "init_model": initialize with another model (see `ìnit_model` parameter)
                * "closed_form": try to initialize with closed form

            Note that unlike in the lrt without partitions, this does not support np.ndarrays.
        :param init_b: (Optional) Low-level initial values for b
            Can be:

            - str:
                * "auto": automatically choose best initialization
                * "standard": initialize with zeros
                * "init_model": initialize with another model (see `ìnit_model` parameter)
                * "closed_form": try to initialize with closed form

            Note that unlike in the lrt without partitions, this does not support np.ndarrays.
        :param size_factors: 1D array of transformed library size factors for each cell in the
            same order as in data
        :param noise_model: str, noise model to use in model-based unit_test. Possible options:

            - 'nb': default
        :param batch_size: The batch size to use for the estimator.
        :param training_strategy: {str, function, list} training strategy to use. Can be:

            - str: will use Estimator.TrainingStrategy[training_strategy] to train
            - function: Can be used to implement custom training function will be called as
              `training_strategy(estimator)`.
            - list of keyword dicts containing method arguments: Will call Estimator.train() once with each dict of
              method arguments.
        :param kwargs: [Debugging] Additional arguments will be passed to the _fit method.
        """
        DETestsSingle = []
        for i, idx in enumerate(self.partition_idx):
            DETestsSingle.append(lrt(
                data=self.x[idx, :],
                reduced_formula_loc=reduced_formula_loc,
                full_formula_loc=full_formula_loc,
                reduced_formula_scale=reduced_formula_scale,
                full_formula_scale=full_formula_scale,
                as_numeric=as_numeric,
                init_a=init_a,
                init_b=init_b,
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
            ave=np.mean(self.x, axis=0),
            correction_type="by_test")

    def wald(
            self,
            factor_loc_totest: str,
            coef_to_test: object = None,  # e.g. coef_to_test="B"
            formula_loc: str = None,
            formula_scale: str = "~1",
            as_numeric: Union[List[str], Tuple[str], str] = (),
            constraints_loc: np.ndarray = None,
            constraints_scale: np.ndarray = None,
            noise_model: str = "nb",
            size_factors: np.ndarray = None,
            batch_size: int = None,
            training_strategy: Union[str, List[Dict[str, object]], Callable] = "AUTO",
            **kwargs
    ):
        """
        Performs a wald test within each partition of a data set.

        See also annotation of de.test.wald()

        :param factor_loc_totest: str, list of strings
            List of factors of formula to test with Wald test.
            E.g. "condition" or ["batch", "condition"] if formula_loc would be "~ 1 + batch + condition"
        :param coef_to_test:
            If there are more than two groups specified by `factor_loc_totest`,
            this parameter allows to specify the group which should be tested.
            Alternatively, if factor_loc_totest is not given, this list sets
            the exact coefficients which are to be tested.
        :param formula_loc: formula
            model formula for location and scale parameter models.
            If not specified, `formula` will be used instead.
        :param formula_scale: formula
            model formula for scale parameter model.
            If not specified, `formula` will be used instead.
        :param as_numeric:
            Which columns of sample_description to treat as numeric and
            not as categorical. This yields columns in the design matrix
            which do not correpond to one-hot encoded discrete factors.
            This makes sense for number of genes, time, pseudotime or space
            for example.
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
        :param batch_size: The batch size to use for the estimator.
        :param training_strategy: {str, function, list} training strategy to use. Can be:

            - str: will use Estimator.TrainingStrategy[training_strategy] to train
            - function: Can be used to implement custom training function will be called as
              `training_strategy(estimator)`.
            - list of keyword dicts containing method arguments: Will call Estimator.train() once with each dict of
              method arguments.
        :param kwargs: [Debugging] Additional arguments will be passed to the _fit method.
        """
        DETestsSingle = []
        for i, idx in enumerate(self.partition_idx):
            DETestsSingle.append(wald(
                data=self.x[idx, :],
                factor_loc_totest=factor_loc_totest,
                coef_to_test=coef_to_test,
                formula_loc=formula_loc,
                formula_scale=formula_scale,
                as_numeric=as_numeric,
                constraints_loc=constraints_loc,
                constraints_scale=constraints_scale,
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
            ave=np.mean(self.x, axis=0),
            correction_type="by_test")


def continuous_1d(
        data: Union[anndata.AnnData, Raw, np.ndarray, scipy.sparse.csr_matrix],
        continuous: str,
        df: int = 5,
        factor_loc_totest: Union[str, List[str]] = None,
        formula_loc: str = None,
        formula_scale: str = "~1",
        as_numeric: Union[List[str], Tuple[str], str] = (),
        test: str = 'wald',
        init_a: Union[np.ndarray, str] = "standard",
        init_b: Union[np.ndarray, str] = "standard",
        gene_names: Union[np.ndarray, list] = None,
        sample_description=None,
        constraints_loc: Union[dict, None] = None,
        constraints_scale: Union[dict, None] = None,
        noise_model: str = 'nb',
        size_factors: np.ndarray = None,
        batch_size: int = None,
        training_strategy: Union[str, List[Dict[str, object]], Callable] = "AUTO",
        quick_scale: bool = None,
        dtype="float64",
        **kwargs
) -> _DifferentialExpressionTestSingle:
    r"""
    Perform differential expression along continous covariate.

    This function wraps the selected statistical test for
    scenarios with continuous covariates and performs the necessary
    spline basis transformation of the continuous co-variate so that the
    problem can be framed as a GLM.

    Note that direct supply of dmats is not enabled as this function wraps
    the building of an adjusted design matrix which contains the spline basis
    covariates. Advanced users who want to control dmat can directly
    perform these spline basis transforms outside of diffxpy and feed the
    dmat directly to one of the test routines wald() or lrt().

    The constraint interface only-supports dictionary-formatted constraints and
    string-formatted constraints but not array-formatted constraint matrices as
    design matrices are built within this function and the shape of constraint
    matrices depends on the output of this function.

    :param data: Array-like or anndata.Anndata object containing observations.
        Input data matrix (observations x features) or (cells x genes).
    :param continuous: str

        - column in data.obs/sample_description which contains the continuous covariate.
    :param df: int
        Degrees of freedom of the spline model, i.e. the number of spline basis vectors.
        df is equal to the number of coefficients in the GLM which are used to describe the
        continuous depedency-
    :param factor_loc_totest:
        List of factors of formula to test with Wald test.
        E.g. "condition" or ["batch", "condition"] if formula_loc would be "~ 1 + batch + condition"
    :param formula_loc: formula
        Model formula for location and scale parameter models.
        If not specified, `formula` will be used instead.
        Refer to continuous covariate by the name givne in the parameter continuous,
        this will be propagated across all coefficients which represent this covariate
        in the spline basis space.
    :param formula_scale: formula
        model formula for scale parameter model.
        If not specified, `formula` will be used instead.
        Refer to continuous covariate by the name givne in the parameter continuous,
        this will be propagated across all coefficients which represent this covariate
        in the spline basis space.
    :param as_numeric:
        Which columns of sample_description to treat as numeric and not as categorical.
        This yields columns in the design matrix which do not correpond to one-hot encoded discrete factors.
        This makes sense for library depth for example. Do not use this for the covariate that you
        want to extrpolate with using a spline-basis!
    :param test: str, statistical test to use. Possible options:

        - 'wald': default
        - 'lrt'
    :param init_a: (Optional) Low-level initial values for a.
        Can be:

        - str:
            * "auto": automatically choose best initialization
            * "standard": initialize intercept with observed mean
        - np.ndarray: direct initialization of 'a'
    :param init_b: (Optional) Low-level initial values for b
        Can be:

        - str:
            * "auto": automatically choose best initialization
            * "standard": initialize with zeros
        - np.ndarray: direct initialization of 'b'
    :param gene_names: optional list/array of gene names which will be used if `data` does
        not implicitly store these
    :param sample_description: optional pandas.DataFrame containing sample annotations
    :param constraints_loc: Constraints for location model. Can be one of the following:

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

        Note that np.ndarray encoded full constraint matrices are not supported here as the design
        matrices are built within this function.
    :param constraints_scale: Constraints for scale model. Can be following:

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

        Note that np.ndarray encoded full constraint matrices are not supported here as the design
        matrices are built within this function.
    :param noise_model: str, noise model to use in model-based unit_test. Possible options:

        - 'nb': default
    :param size_factors: 1D array of transformed library size factors for each cell in the
        same order as in data
    :param batch_size: the batch size to use for the estimator
    :param training_strategy: {str, function, list} training strategy to use. Can be:

        - str: will use Estimator.TrainingStrategy[training_strategy] to train
        - function: Can be used to implement custom training function will be called as
            `training_strategy(estimator)`.
        - list of keyword dicts containing method arguments: Will call Estimator.train()
            once with each dict of method arguments.

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
    if formula_loc is None:
        raise ValueError("supply fomula_loc")
    # Set testing default to continuous covariate if not supplied:
    if factor_loc_totest is None:
        factor_loc_totest = [continuous]
    elif isinstance(factor_loc_totest, str):
        factor_loc_totest = [factor_loc_totest]
    elif isinstance(factor_loc_totest, tuple):
        factor_loc_totest = list(factor_loc_totest)

    if isinstance(as_numeric, str):
        as_numeric = [as_numeric]
    if isinstance(as_numeric, tuple):
        as_numeric = list(as_numeric)

    gene_names = parse_gene_names(data, gene_names)
    sample_description = parse_sample_description(data, sample_description)

    # Check that continuous factor is contained in sample description
    if continuous not in sample_description.columns:
        raise ValueError('parameter continuous not found in sample_description')

    # Perform spline basis transform.
    spline_basis = patsy.highlevel.dmatrix("0+bs(" + continuous + ", df=" + str(df) + ")", sample_description)
    spline_basis = pd.DataFrame(spline_basis)
    new_coefs = [continuous + str(i) for i in range(spline_basis.shape[1])]
    spline_basis.columns = new_coefs
    formula_extension = '+'.join(new_coefs)
    # Generated interpolated spline bases.
    # Safe interpolated interval in last column, need to extract later.
    interpolated_interval = np.linspace(
        np.min(sample_description[continuous].values),
        np.max(sample_description[continuous].values),
        100
    )
    interpolated_spline_basis = np.hstack([
        np.ones([100, 1]),
        patsy.highlevel.dmatrix(
            "0+bs(" + continuous + ", df=" + str(df) + ")",
            pd.DataFrame({continuous: interpolated_interval})
        ).base,
        np.expand_dims(interpolated_interval, axis=1)
    ])

    # Replace continuous factor in formulas by spline basis coefficients.
    # Note that the brackets around formula_term_continuous propagate the sum
    # across interaction terms.
    formula_term_continuous = '(' + formula_extension + ')'

    if formula_loc is not None:
        formula_loc_new = formula_loc.split(continuous)
        formula_loc_new = formula_term_continuous.join(formula_loc_new)
    else:
        formula_loc_new = None

    if formula_scale is not None:
        formula_scale_new = formula_scale.split(continuous)
        formula_scale_new = formula_term_continuous.join(formula_scale_new)
    else:
        formula_scale_new = None

    # Add spline basis into sample description
    for x in spline_basis.columns:
        sample_description[x] = spline_basis[x].values

    # Add spline basis to continuous covariate list
    as_numeric.extend(new_coefs)

    if test.lower() == 'wald':
        if noise_model is None:
            raise ValueError("Please specify noise_model")

        # Adjust factors / coefficients to test:
        # Note that the continuous covariate does not necessarily have to be tested,
        # it could also be a condition effect or similar.
        # TODO handle interactions
        if continuous in factor_loc_totest:
            # Create reduced set of factors to test which does not contain continuous:
            factor_loc_totest_new = [x for x in factor_loc_totest if x != continuous]
            # Add spline basis terms in instead of continuous term:
            factor_loc_totest_new.extend(new_coefs)
        else:
            factor_loc_totest_new = factor_loc_totest

        logging.getLogger("diffxpy").debug("model formulas assembled in de.test.continuos():")
        logging.getLogger("diffxpy").debug("factor_loc_totest_new: " + ",".join(factor_loc_totest_new))
        logging.getLogger("diffxpy").debug("formula_loc_new: " + formula_loc_new)
        logging.getLogger("diffxpy").debug("formula_scale_new: " + formula_scale_new)

        de_test = wald(
            data=data,
            factor_loc_totest=factor_loc_totest_new,
            coef_to_test=None,
            formula_loc=formula_loc_new,
            formula_scale=formula_scale_new,
            as_numeric=as_numeric,
            init_a=init_a,
            init_b=init_b,
            gene_names=gene_names,
            sample_description=sample_description,
            constraints_loc=constraints_loc,
            constraints_scale=constraints_scale,
            noise_model=noise_model,
            size_factors=size_factors,
            batch_size=batch_size,
            training_strategy=training_strategy,
            quick_scale=quick_scale,
            dtype=dtype,
            **kwargs
        )
        de_test = DifferentialExpressionTestWaldCont(
            de_test=de_test,
            noise_model=noise_model,
            size_factors=size_factors,
            continuous_coords=sample_description[continuous].values,
            spline_coefs=new_coefs,
            interpolated_spline_basis=interpolated_spline_basis
        )
    elif test.lower() == 'lrt':
        if noise_model is None:
            raise ValueError("Please specify noise_model")
        full_formula_loc = formula_loc_new
        # Assemble reduced loc model:
        formula_scale_new = formula_scale.split(continuous)
        formula_scale_new = formula_term_continuous.join(formula_scale_new)
        reduced_formula_loc = formula_scale.split('+')
        # Take out terms in reduced location model which are to be tested:
        reduced_formula_loc = [x for x in reduced_formula_loc if x not in factor_loc_totest]
        reduced_formula_loc = '+'.join(reduced_formula_loc)
        # Replace occurences of continuous term in reduced model:
        reduced_formula_loc = reduced_formula_loc.split(continuous)
        reduced_formula_loc = formula_term_continuous.join(reduced_formula_loc)

        # Scale model is not tested:
        full_formula_scale = formula_scale_new
        reduced_formula_scale = formula_scale_new

        logging.getLogger("diffxpy").debug("model formulas assembled in de.test.continuous():")
        logging.getLogger("diffxpy").debug("full_formula_loc: " + full_formula_loc)
        logging.getLogger("diffxpy").debug("reduced_formula_loc: " + reduced_formula_loc)
        logging.getLogger("diffxpy").debug("full_formula_scale: " + full_formula_scale)
        logging.getLogger("diffxpy").debug("reduced_formula_scale: " + reduced_formula_scale)

        de_test = lrt(
            data=data,
            full_formula_loc=full_formula_loc,
            reduced_formula_loc=reduced_formula_loc,
            full_formula_scale=full_formula_scale,
            reduced_formula_scale=reduced_formula_scale,
            as_numeric=as_numeric,
            init_a=init_a,
            init_b=init_b,
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
        de_test = DifferentialExpressionTestLRTCont(
            de_test=de_test,
            size_factors=size_factors,
            continuous_coords=sample_description[continuous].values,
            spline_coefs=new_coefs,
            noise_model=noise_model
        )
    else:
        raise ValueError('continuous(): Parameter `test` not recognized.')

    return de_test
