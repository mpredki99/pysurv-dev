import numpy as np

from ...constants import INVALID_INDEX
from .inner_constraints_builder import InnerConstraintsBuilder


class MatrixSXBuilder(InnerConstraintsBuilder):
    def __init__(self, dataset, matrix_x_indexer, matrix_x_n_col):
        super().__init__(dataset, matrix_x_indexer, matrix_x_n_col)

    def build(self):
        sx = np.zeros(self._matrix_x_n_col)

        coord_idx = self._matrix_x_indexer.coordinates_indices.values.flatten()
        coord_idx = coord_idx[coord_idx != INVALID_INDEX]
        sx[coord_idx] = 1

        return sx
