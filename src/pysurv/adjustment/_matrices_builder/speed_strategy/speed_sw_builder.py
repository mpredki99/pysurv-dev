from ..matrix_sw_builder import MatrixSWBuilder
from ...constants import INVALID_INDEX


class SpeedSWBuilder(MatrixSWBuilder):
    def __init__(self, dataset, matrix_x_indexer, default_sigmas):
        super().__init__(dataset, matrix_x_indexer, default_sigmas)

    def build(self, matrix_x_n_col):
        sW = self._initialize_sw_matrix(matrix_x_n_col)

        coord_sigmas = self._dataset.controls.coordinates_sigma.copy()
        coord_indices = self._prepare_coord_indices()

        self._fill_coord_sigmas(coord_sigmas)
        coord_sigmas = self._melt_coord_sigmas(coord_sigmas)

        data = coord_indices.merge(
            coord_sigmas,
            how="left",
            on=["id", "coord"],
        )

        for row in data.itertuples(index=False):
            idx = getattr(row, "idx")
            sigma_value = getattr(row, "sigma_value")
            sW[idx] = 1 / (sigma_value ** 2) if sigma_value > 0 else 0

        return sW

    def _prepare_coord_indices(self):
        coord_columns = self._dataset.controls.coordinates_columns
        indices = self._matrix_x_indexer.coordinates_indices.reset_index()
        indices = indices.melt(
            id_vars="id", value_vars=coord_columns, var_name="coord", value_name="idx"
        )
        return indices[indices["idx"] != INVALID_INDEX]

    def _fill_coord_sigmas(self, coord_sigmas):
        for col in self._dataset.controls.coordinates_columns:
            sigma_col = f"s{col}"
            if sigma_col in coord_sigmas:
                coord_sigmas[col] = coord_sigmas[sigma_col].fillna(
                    self._default_sigmas[sigma_col]
                )
                coord_sigmas.drop(columns=sigma_col, inplace=True)
            else:
                coord_sigmas[col] = self._default_sigmas[sigma_col]

    def _melt_coord_sigmas(self, sigmas):
        sigmas = sigmas.reset_index()
        sigmas = sigmas.melt(
            id_vars="id",
            value_vars=sigmas.columns,
            var_name="coord",
            value_name="sigma_value",
        )
        return sigmas
