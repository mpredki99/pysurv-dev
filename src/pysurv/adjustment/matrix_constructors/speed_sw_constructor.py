# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

import numpy as np
import pandas as pd

from pysurv.data.dataset import Dataset

from .._constants import INVALID_INDEX
from .indexer_matrix_x import IndexerMatrixX
from .matrix_base_constructors import MatrixSWConstructor


class SpeedSWConstructor(MatrixSWConstructor):
    """
    SpeedSWBuilder constructs the control point weights matrix (sW) for the coordinates using fast, vectorized approach.
    """

    def __init__(
        self,
        dataset: Dataset,
        matrix_x_indexer: IndexerMatrixX,
        default_sigmas_index: str | None,
    ):
        super().__init__(dataset, matrix_x_indexer, default_sigmas_index)

    def build(self, matrix_x_n_col: int) -> np.ndarray:
        """Builds and returns the diagonal controls weight matrix (sW)."""
        coord_sigmas = self._dataset.controls.coordinate_sigmas.copy()

        sW = self._initialize_sw_matrix(matrix_x_n_col)

        coord_indices = self._melt_coord_indices()
        coord_sigmas = self._fill_coord_sigmas(coord_sigmas)
        coord_sigmas = self._melt_coord_sigmas(coord_sigmas)

        coord_data = coord_indices.merge(
            coord_sigmas,
            how="left",
            on=["id", "coord"],
        )

        for row in coord_data.itertuples(index=False):
            idx = getattr(row, "idx")
            sigma_value = getattr(row, "sigma_value")
            sW[idx] = 1 / (sigma_value**2) if sigma_value > 0 else 0

        return np.diag(sW)

    def _melt_coord_indices(self) -> pd.DataFrame:
        """Reshape control point indices from wide to long format."""
        coordinate_columns = self._dataset.controls.coordinate_columns
        indices = self._matrix_x_indexer.coordinate_indices.reset_index()
        indices = indices.melt(
            id_vars="id",
            value_vars=coordinate_columns,
            var_name="coord",
            value_name="idx",
        )
        return indices[indices["idx"] != INVALID_INDEX]

    def _fill_coord_sigmas(self, coord_sigmas: pd.DataFrame) -> pd.DataFrame:
        """Fill missing coordinate sigmas with default values."""
        coordinate_columns = self._dataset.controls.coordinate_columns
        for coord_col in coordinate_columns:
            sigma_col = f"s{coord_col}"
            if sigma_col in coord_sigmas:
                coord_sigmas[coord_col] = coord_sigmas[sigma_col].fillna(
                    self._default_sigmas[sigma_col]
                )
                coord_sigmas.drop(columns=sigma_col, inplace=True)
            else:
                coord_sigmas[coord_col] = self._default_sigmas[sigma_col]
        return coord_sigmas

    def _melt_coord_sigmas(self, coord_sigmas: pd.DataFrame) -> pd.DataFrame:
        """Reshape coordinate sigmas DataFrame from wide to long format."""
        coord_sigmas = coord_sigmas.reset_index()
        coord_sigmas = coord_sigmas.melt(
            id_vars="id",
            value_vars=coord_sigmas.columns,
            var_name="coord",
            value_name="sigma_value",
        )
        return coord_sigmas
