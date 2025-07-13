import numpy as np
import pandas as pd


class MemoryXYWBuilder:
    def __init__(self, parent):
        self._parent = parent
        
    def build_xyw(self, calculate_weights):
        measurements = self._parent.dataset.measurements
        controls = self._parent.dataset.controls
        stations = self._parent.dataset.stations

        stn_pk_loc = measurements.index.names.index("stn_pk")
        trg_id_loc = measurements.index.names.index("trg_id")
        trg_h_loc = (
            measurements.index.names.index("trg_h")
            if "trg_h" in measurements.index.names
            else None
        )

        X, Y, W = self._parent.initialize_xyw_matrices(calculate_weights)

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

                matrix_x_col_indices = {
                    f"{coord}_{label}": self._parent.cordinates_index_in_x_matrix.at[
                        ctrl_id, coord
                    ]
                    for ctrl_id, label in zip([stn_id, trg_id], ["stn", "trg"])
                    for coord in controls.coordinates.columns
                }
                if meas_type == "hz":
                    coord_diff["orientation"] = stations.orientation[stn_pk]
                    matrix_x_col_indices["orientation_idx"] = (
                        self._parent.orientations_index_in_x_matrix[stn_pk]
                    )

                self._parent.apply_observation_function(
                    meas_type,
                    meas_value,
                    coord_diff,
                    matrix_x_col_indices,
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
        sigma_value = None
        if sigma_type in self._parent.dataset.measurements.sigma_columns:
            sigma_value = self._parent.dataset.measurements.at[idx, sigma_type]

        return (
            sigma_value
            if not pd.isna(sigma_value)
            else self._parent.default_sigmas[sigma_type]
        )