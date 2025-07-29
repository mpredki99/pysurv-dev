# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

import numpy as np
import pandas as pd

from pysurv.data.dataset import Dataset

from .._constants import INVALID_INDEX
from ._matrix_sw_constructor import MatrixSWConstructor
from ._matrix_x_indexer import MatrixXIndexer


class MemorySWConstructor(MatrixSWConstructor):
    """
    Constructs the control point weights matrix (sW) for the coordinates using a memory-efficient,
    row-wise approach.
    """

    def __init__(
        self,
        dataset: Dataset,
        matrix_x_indexer: MatrixXIndexer,
        default_sigmas: pd.Series,
    ):
        super().__init__(dataset, matrix_x_indexer, default_sigmas)

    def build(self, matrix_x_n_col: int) -> np.ndarray:
        """Builds and returns the diagonal controls weight matrix (sW) in a memory-efficient way."""
        sW = self._initialize_sw_matrix(matrix_x_n_col)

        coordinate_indices = self._matrix_x_indexer.coordinate_indices

        for idx in coordinate_indices.index:
            for coord in coordinate_indices.columns:
                sw_idx = coordinate_indices.at[idx, coord]
                if sw_idx == INVALID_INDEX:
                    continue
                sigma_value = self._get_coord_sigma_value(idx, f"s{coord}")
                sW[sw_idx] = 1 / (sigma_value**2) if sigma_value > 0 else 0

        return np.diag(sW)

    def _get_coord_sigma_value(self, idx: pd.Index, sigma_coord: str):
        """Return the sigma value for a given control point sigma column."""
        coord_columns = self._dataset.controls.columns
        sigma_value = None
        if sigma_coord in coord_columns:
            sigma_value = self._dataset.controls.at[idx, sigma_coord]
        if sigma_value is None:
            return self._default_sigmas[sigma_coord]
        return sigma_value
