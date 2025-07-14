import numpy as np

from ..matrix_builder import MatrixBuilder


class InnerConstraintsBuilder(MatrixBuilder):
    def __init__(self, dataset, matrix_x_indexer, matrix_x_n_col):
        super().__init__(dataset, matrix_x_indexer)
        self._matrix_x_n_col = matrix_x_n_col
