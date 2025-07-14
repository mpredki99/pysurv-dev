from ..sigma_config import sigma_config
from .matrix_builder import MatrixBuilder


class MatrixXYWsWBuilder(MatrixBuilder):
    def __init__(self, dataset, matrix_x_indexer, default_sigmas):
        super().__init__(dataset, matrix_x_indexer)

        default_sigmas = default_sigmas or sigma_config.default_index
        self._default_sigmas = sigma_config[default_sigmas]
