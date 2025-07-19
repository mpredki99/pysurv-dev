# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

import numpy as np
import pandas as pd

from pysurv.data.dataset import Dataset

from .._constants import INVALID_INDEX


class MatrixXIndexer:
    """
    Indexer for mapping control point coordinates and station orientations to matrix X column indices.

    This class provides methods to lazy generation and access index mappings for control point coordinates
    and station orientations, which are used as columns in the design matrix (X)for least squares
    adjustment computations.
    """

    def __init__(self, dataset: Dataset) -> None:
        self._controls = dataset.controls
        self._stations = dataset.stations

        self._coordinate_indices = None
        self._orientation_indices = None

    @property
    def coordinate_indices(self) -> pd.DataFrame:
        """Return a DataFrame mapping control points to their columns in matrix X."""
        if self._coordinate_indices is None:
            self._map_coordinate_index()
        return self._coordinate_indices

    @property
    def orientation_indices(self) -> pd.Series:
        """Return a Series mapping station orientations to their columns in matrix X."""
        if self._orientation_indices is None:
            if self._coordinate_indices is None:
                self._map_coordinate_index()
            self._map_orientation_index()
        return self._orientation_indices

    def _map_coordinate_index(self) -> None:
        """Map control point coordinates to their corresponding column indices in matrix X."""
        coord_idx = self._controls.index
        coord_columns = self._controls.coordinate_columns

        coord_values = self._controls.coordinates.values
        mask = np.isfinite(coord_values)

        index_map = np.full(coord_values.shape, INVALID_INDEX, dtype=int)
        index_map[mask] = np.arange(np.count_nonzero(mask))

        self._coordinate_indices = pd.DataFrame(
            index_map, index=coord_idx, columns=coord_columns
        )

    def _map_orientation_index(self) -> None:
        """Map station orientations to their corresponding column indices in matrix X."""
        orientations = self._stations.orientation.dropna()

        start_idx = self._coordinate_indices.max().max() + 1
        end_idx = start_idx + len(orientations)

        index_values = np.arange(start_idx, end_idx)

        self._orientation_indices = pd.Series(
            index_values, index=orientations.index, name="orientation_idx"
        )
