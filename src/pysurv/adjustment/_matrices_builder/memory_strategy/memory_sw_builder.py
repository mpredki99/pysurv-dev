from ..matrix_sw_builder import MatrixSWBuilder
from ...constants import INVALID_INDEX


class MemorySWBuilder(MatrixSWBuilder):
    def __init__(self, dataset, matrix_x_indexer, default_sigmas):
        super().__init__(dataset, matrix_x_indexer, default_sigmas)

    def build(self, matrix_x_n_col):
        coord_indices = self._matrix_x_indexer.coordinates_indices
        sW = self._initialize_sw_matrix(matrix_x_n_col)

        for idx in coord_indices.index:
            for coord in coord_indices.columns:
                sw_idx = coord_indices.at[idx, coord]
                if sw_idx == INVALID_INDEX:
                    continue
                sigma_value = self._get_coord_sigma_value(idx, f"s{coord}")
                sW[sw_idx] = 1 / (sigma_value**2) if sigma_value > 0 else 0

        return sW

    def _get_coord_sigma_value(self, idx, scoord):
        if scoord in self._dataset.controls.columns:
            return self._dataset.controls.at[idx, scoord]
        else:
            return self._default_sigmas[scoord]
