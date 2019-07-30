import anndata
import numpy as np
import pandas as pd
import patsy
import scipy
from typing import List, Tuple, Union
import xarray as xr

try:
    from anndata.base import Raw
except ImportError:
    from anndata import Raw

from batchglm import data as data_utils
# Relay util functions for diffxpy api.
# design_matrix, preview_coef_names and constraint_system_from_star are redefined here.
from batchglm.data import constraint_matrix_from_string, constraint_matrix_from_dict
from batchglm.data import design_matrix_from_xarray, design_matrix_from_anndata
from batchglm.data import view_coef_names


def parse_gene_names(data, gene_names):
    if gene_names is None:
        if anndata is not None and (isinstance(data, anndata.AnnData) or isinstance(data, Raw)):
            gene_names = data.var_names
        elif isinstance(data, xr.DataArray):
            gene_names = data["features"]
        elif isinstance(data, xr.Dataset):
            gene_names = data["features"]
        else:
            raise ValueError("Missing gene names")

    return np.asarray(gene_names)


def parse_data(data, gene_names) -> xr.DataArray:
    X = data_utils.xarray_from_data(data, dims=("observations", "features"))
    if gene_names is not None:
        X.coords["features"] = gene_names

    return X


def parse_sample_description(
        data: Union[anndata.AnnData, Raw, xr.DataArray, xr.Dataset, np.ndarray, scipy.sparse.csr_matrix],
        sample_description: Union[pd.DataFrame, None]
) -> pd.DataFrame:
    """
    Parse sample description from input.

    :param data: Input data matrix (observations x features) or (cells x genes).
    :param sample_description: pandas.DataFrame containing sample annotations, can be None.
    :return: Assembled sample annotations.
    """
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

    if anndata is not None and isinstance(data, Raw):
        # Raw does not have attribute shape.
        assert data.X.shape[0] == sample_description.shape[0], \
            "data matrix and sample description must contain same number of cells"
    else:
        assert data.shape[0] == sample_description.shape[0], \
            "data matrix and sample description must contain same number of cells"
    return sample_description


def parse_size_factors(
        size_factors: Union[np.ndarray, pd.core.series.Series, np.ndarray],
        data: Union[anndata.AnnData, Raw, xr.DataArray, xr.Dataset, np.ndarray, scipy.sparse.csr_matrix],
        sample_description: pd.DataFrame
) -> Union[np.ndarray, None]:
    """
    Parse size-factors from input.

    :param size_factors: 1D array of transformed library size factors for each cell in the
        same order as in data or string-type column identifier of size-factor containing
        column in sample description.
    :param data: Input data matrix (observations x features) or (cells x genes).
    :param sample_description: optional pandas.DataFrame containing sample annotations
    :return: Assebled size-factors.
    """
    if size_factors is not None:
        if isinstance(size_factors, pd.core.series.Series):
            size_factors = size_factors.values
        elif isinstance(size_factors, str):
            assert size_factors in sample_description.columns, ""
            size_factors = sample_description[size_factors].values
        assert size_factors.shape[0] == data.shape[0], "data matrix and size factors must contain same number of cells"
        assert np.all(size_factors > 0), "size_factors <= 0 found, please remove these cells"
    return size_factors


def parse_grouping(data, sample_description, grouping):
    if isinstance(grouping, str):
        sample_description = parse_sample_description(data, sample_description)
        grouping = sample_description[grouping]
    return np.squeeze(np.asarray(grouping))


def split_X(data, grouping):
    groups = np.unique(grouping)
    x0 = data[np.where(grouping == groups[0])[0]]
    x1 = data[np.where(grouping == groups[1])[0]]
    return x0, x1


def dmat_unique(dmat, sample_description):
    dmat, idx = np.unique(dmat, axis=0, return_index=True)
    sample_description = sample_description.iloc[idx].reset_index(drop=True)

    return dmat, sample_description


def design_matrix(
        data: Union[anndata.AnnData, Raw, xr.DataArray, xr.Dataset, np.ndarray,
                    scipy.sparse.csr_matrix] = None,
        sample_description: Union[None, pd.DataFrame] = None,
        formula: Union[None, str] = None,
        as_numeric: Union[List[str], Tuple[str], str] = (),
        dmat: Union[pd.DataFrame, None] = None,
        return_type: str = "xarray"
) -> Union[patsy.design_info.DesignMatrix, xr.Dataset, pd.DataFrame]:
    """ Create a design matrix from some sample description.

    This function defaults to perform formatting if dmat is directly supplied as a pd.DataFrame.
    This function relays batchglm.data.design_matrix() to behave like the other wrappers in diffxpy.

    :param data: Input data matrix (observations x features) or (cells x genes).
    :param sample_description: pandas.DataFrame of length "num_observations" containing explanatory variables as columns
    :param formula: model formula as string, describing the relations of the explanatory variables.

        E.g. '~ 1 + batch + confounder'
    :param as_numeric:
        Which columns of sample_description to treat as numeric and
        not as categorical. This yields columns in the design matrix
        which do not correpond to one-hot encoded discrete factors.
        This makes sense for number of genes, time, pseudotime or space
        for example.
    :param dmat: a model design matrix as a pd.DataFrame
    :param return_type: type of the returned value.

        - "patsy": return plain patsy.design_info.DesignMatrix object
        - "dataframe": return pd.DataFrame with observations as rows and params as columns
        - "xarray": return xr.Dataset with design matrix as ds["design"] and the sample description embedded as
            one variable per column
    :param dmat: model design matrix
    """
    if data is None and sample_description is None and dmat is None:
        raise ValueError("supply either data or sample_description or dmat")
    if dmat is None and formula is None:
        raise ValueError("supply either dmat or formula")

    if dmat is None:
        sample_description = parse_sample_description(data, sample_description)

    if sample_description is not None:
        as_categorical = [False if x in as_numeric else True for x in sample_description.columns.values]
    else:
        as_categorical = True

    return data_utils.design_matrix(
        sample_description=sample_description,
        formula=formula,
        as_categorical=as_categorical,
        dmat=dmat,
        return_type=return_type
    )


def preview_coef_names(
        sample_description: pd.DataFrame,
        formula: str,
        as_numeric: Union[List[str], Tuple[str], str] = ()
) -> np.ndarray:
    """
    Return coefficient names of model.

    Use this to preview what the model would look like.
    This function relays batchglm.data.preview_coef_names() to behave like the other wrappers in diffxpy.

    :param sample_description: pandas.DataFrame of length "num_observations" containing explanatory variables as columns
    :param formula: model formula as string, describing the relations of the explanatory variables.

        E.g. '~ 1 + batch + confounder'
    :param as_numeric:
        Which columns of sample_description to treat as numeric and
        not as categorical. This yields columns in the design matrix
        which do not correpond to one-hot encoded discrete factors.
        This makes sense for number of genes, time, pseudotime or space
        for example.
    :return: A list of coefficient names.
    """
    if isinstance(as_numeric, str):
        as_numeric = [as_numeric]
    if isinstance(as_numeric, tuple):
        as_numeric = list(as_numeric)

    return data_utils.preview_coef_names(
        sample_description=sample_description,
        formula=formula,
        as_categorical=[False if x in as_numeric else True for x in sample_description.columns.values]
    )


def constraint_system_from_star(
        dmat: Union[None, np.ndarray, xr.DataArray, xr.Dataset] = None,
        sample_description: Union[None, pd.DataFrame] = None,
        formula: Union[None, str] = None,
        as_numeric: Union[List[str], Tuple[str], str] = (),
        constraints: dict = {},
        dims: Union[Tuple[str, str], List[str]] = (),
        return_type: str = "xarray",
) -> Tuple:
    """
    Create a design matrix and a constraint matrix.

    This function relays batchglm.data.constraint_matrix_from_star() to behave like the other wrappers in diffxpy.

    :param dmat: Pre-built model design matrix.
    :param sample_description: pandas.DataFrame of length "num_observations" containing explanatory variables as columns
    :param formula: model formula as string, describing the relations of the explanatory variables.

        E.g. '~ 1 + batch + confounder'
    :param as_numeric:
        Which columns of sample_description to treat as numeric and
        not as categorical. This yields columns in the design matrix
        which do not correspond to one-hot encoded discrete factors.
    :param constraints: Grouped factors to enfore equality constraints on. Every element of
        the dictionary corresponds to one set of equality constraints. Each set has to be
        be an entry of the form {..., x: y, ...} where x is the factor to be constrained and y is
        a factor by which levels of x are grouped and then constrained. Set y="1" to constrain
        all levels of x to sum to one, a single equality constraint.

            E.g.: {"batch": "condition"} Batch levels within each condition are constrained to sum to
                zero. This is applicable if repeats of a an experiment within each condition
                are independent so that the set-up ~1+condition+batch is perfectly confounded.

        Can only group by non-constrained effects right now, use constraint_matrix_from_string
        for other cases.
    :param dims: Dimension names of xarray.

        E.g.: ["design_loc_params", "loc_params"] or ["design_scale_params", "scale_params"]
    :param return_type: type of the returned value.

        - "patsy": return plain patsy.design_info.DesignMatrix object
        - "dataframe": return pd.DataFrame with observations as rows and params as columns
        - "xarray": return xr.Dataset with design matrix as ds["design"] and the sample description embedded as
            one variable per column
        This option is overridden if constraints are supplied as dict.
    :return: a model design matrix and a constraint matrix formatted as xr.DataArray
    """
    if isinstance(as_numeric, str):
        as_numeric = [as_numeric]
    if isinstance(as_numeric, tuple):
        as_numeric = list(as_numeric)

    if sample_description is not None:
        as_categorical = [False if x in as_numeric else True for x in sample_description.columns.values]
    else:
        as_categorical = True

    return data_utils.constraint_system_from_star(
        dmat=dmat,
        sample_description=sample_description,
        formula=formula,
        as_categorical=as_categorical,
        constraints=constraints,
        dims=dims,
        return_type=return_type
    )
