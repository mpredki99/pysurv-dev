# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

import numpy as np

from pysurv.data.dataset import Dataset

from .._constants import INVALID_INDEX
from ._inner_constraints import InnerConstraintsConstructor
from ._matrix_x_indexer import MatrixXIndexer


class MatrixSXConstructor(InnerConstraintsConstructor):
    """Constructs the control point corrections matrix (sX) for adjustment computations."""

    def __init__(
        self, dataset: Dataset, matrix_x_indexer: MatrixXIndexer, matrix_x_n_col: int
    ) -> None:
        super().__init__(dataset, matrix_x_indexer, matrix_x_n_col)

    def build(self):
        """Build and return the control point corrections matrix (sX)."""
        sx = np.zeros(self._matrix_x_n_col)

        coord_idx = self._matrix_x_indexer.coordinate_indices.values.flatten()
        coord_idx = coord_idx[coord_idx != INVALID_INDEX]
        sx[coord_idx] = 1

        return sx
