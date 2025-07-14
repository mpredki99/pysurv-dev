from abc import abstractmethod

import numpy as np

from .matrix_xyw_sw_builder import MatrixXYWsWBuilder


class MatrixSWBuilder(MatrixXYWsWBuilder):
    def __init__(self, dataset, matrix_x_indexer, default_sigmas):
        super().__init__(dataset, matrix_x_indexer, default_sigmas)

    @abstractmethod
    def build(self, matrix_x_n_col):
        pass

    def _initialize_sw_matrix(self, matrix_x_n_col):
        return np.zeros(matrix_x_n_col)
