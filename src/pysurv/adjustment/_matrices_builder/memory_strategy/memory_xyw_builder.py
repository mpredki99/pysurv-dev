import numpy as np

from ..matrix_xyw_builder import MatrixXYWBuilder


class MemoryXYWBuilder(MatrixXYWBuilder):
    def __init__(self, dataset, matrix_x_indexer, default_sigmas):
        super().__init__(dataset, matrix_x_indexer, default_sigmas)

    def build(self, calculate_weights):
        measurements = self._dataset.measurements
        controls = self._dataset.controls
        stations = self._dataset.stations

        coord_index = self._matrix_x_indexer.coordinates_indices
        orientation_index = self._matrix_x_indexer.orientations_indices

        stn_pk_loc = measurements.index.names.index("stn_pk")
        trg_id_loc = measurements.index.names.index("trg_id")
        trg_h_loc = (
            measurements.index.names.index("trg_h")
            if "trg_h" in measurements.index.names
            else None
        )

        X, Y, W = self._initialize_xyw_matrices(calculate_weights)

        eq_row = 0
        for idx in measurements.index:
            stn_pk = idx[stn_pk_loc]
            stn_id = stations.at[stn_pk, "stn_id"]
            stn_h = stations.at[stn_pk, "stn_h"] if "stn_h" in stations.columns else 0

            trg_id = idx[trg_id_loc]
            trg_h = idx[trg_h_loc] if trg_h_loc else 0

            coord_diff = {
                f"d{col}": controls.at[trg_id, col] - controls.at[stn_id, col]
                for col in controls.coordinates_columns
                if col != "z"
            }
            if "z" in controls.columns:
                coord_diff["dz"] = (
                    controls.at[trg_id, "z"] + trg_h - controls.at[stn_id, "z"] - stn_h
                )

            for meas_type in measurements.measurements_columns:
                meas_value = measurements.at[idx, meas_type]
                if np.isnan(meas_value):
                    continue

                matrix_x_col_index = {
                    f"{coord}_{label}": coord_index.at[ctrl_id, coord]
                    for ctrl_id, label in zip([stn_id, trg_id], ["stn", "trg"])
                    for coord in controls.coordinates_columns
                }
                if meas_type == "hz":
                    coord_diff["orientation"] = stations.orientation[stn_pk]
                    matrix_x_col_index["orientation_idx"] = orientation_index[stn_pk]
                self._apply_observation_function(
                    meas_type,
                    meas_value,
                    coord_diff,
                    matrix_x_col_index,
                    X[eq_row, :],
                    eq_row,
                    Y,
                )

                if calculate_weights:
                    sigma_value = self._get_sigma_value(idx, meas_type)
                    W[eq_row] = 1 / sigma_value**2

                eq_row += 1

        return X, Y, np.diag(W) if W is not None else None

    def _get_sigma_value(self, idx, meas_type):
        sigma_type = f"s{meas_type}"
        if sigma_type in self._dataset.measurements.sigma_columns:
            return self._dataset.measurements.at[idx, sigma_type]
        else:
            return self._default_sigmas[sigma_type]
