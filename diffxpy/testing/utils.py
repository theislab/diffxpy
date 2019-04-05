from typing import Union

import anndata
import numpy as np
import pandas as pd
import patsy
import xarray as xr

from batchglm import data as data_utils


def parse_gene_names(data, gene_names):
    if gene_names is None:
        if anndata is not None and (isinstance(data, anndata.AnnData) or isinstance(data, anndata.base.Raw)):
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


def parse_sample_description(data, sample_description=None) -> pd.DataFrame:
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

    if anndata is not None and isinstance(data, anndata.base.Raw):
        # anndata.base.Raw does not have attribute shape.
        assert data.X.shape[0] == sample_description.shape[0], \
            "data matrix and sample description must contain same number of cells"
    else:
        assert data.shape[0] == sample_description.shape[0], \
            "data matrix and sample description must contain same number of cells"
    return sample_description


def parse_size_factors(size_factors, data):
    if size_factors is not None:
        if isinstance(size_factors, pd.core.series.Series):
            size_factors = size_factors.values
        assert size_factors.shape[0] == data.shape[0], "data matrix and size factors must contain same number of cells"
        assert np.all(size_factors > 0), "size_factors <= 0 found, please remove these cells"
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
        sample_description = parse_sample_description(data, sample_description)
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