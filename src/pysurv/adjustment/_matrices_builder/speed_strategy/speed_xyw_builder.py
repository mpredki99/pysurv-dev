import numpy as np

from ..matrix_xyw_builder import MatrixXYWBuilder


class SpeedXYWBuilder(MatrixXYWBuilder):
    def __init__(self, dataset, matrix_x_indexer, default_sigmas):
        super().__init__(dataset, matrix_x_indexer, default_sigmas)

    def build(self, calculate_weights):
        stations_columns = self._dataset.stations.columns

        data, idx_names = self._prepare_subset(
            self._dataset.measurements.measurement_values
        )
        data = self._melt_subset(data, idx_names, "meas")
        data.dropna(subset="meas_value", inplace=True)

        if calculate_weights:
            sigmas, _ = self._prepare_subset(self._dataset.measurements.sigma_values)
            self._fill_sigmas(sigmas, self._dataset.measurements.measurements_columns)
            sigmas = self._melt_subset(sigmas, idx_names, "sigma")

            on_cols = idx_names.append("meas_type")
            data = data.merge(sigmas, how="left", on=on_cols)

        data = self._merge_with_controls(data, self._dataset.controls.coordinates)
        self._calculate_coord_differences(data)
        data.drop(
            columns=["x_stn", "y_stn", "z_stn", "x_trg", "y_trg", "z_trg"],
            inplace=True,
            errors="ignore",
        )
        data = self._merge_with_controls(
            data, self._matrix_x_indexer.coordinates_indices
        )

        if "orientation" in stations_columns:
            data = self._merge_orientations(data)

        X, Y, W = self._initialize_xyw_matrices(calculate_weights)

        for eq_row, row in enumerate(data.itertuples(index=False)):
            coord_diff = {
                "dx": getattr(row, "dx", None),
                "dy": getattr(row, "dy", None),
                "dz": getattr(row, "dz", None),
                "orientation": getattr(row, "orientation", None),
            }
            matrix_x_col_indices = {
                "x_stn": getattr(row, "x_stn", None),
                "y_stn": getattr(row, "y_stn", None),
                "z_stn": getattr(row, "z_stn", None),
                "x_trg": getattr(row, "x_trg", None),
                "y_trg": getattr(row, "y_trg", None),
                "z_trg": getattr(row, "z_trg", None),
                "orientation_idx": getattr(row, "orientation_idx", None),
            }
            self._apply_observation_function(
                getattr(row, "meas_type"),
                getattr(row, "meas_value"),
                coord_diff,
                matrix_x_col_indices,
                X[eq_row, :],
                eq_row,
                Y,
            )

            if calculate_weights:
                W[eq_row] = 1 / getattr(row, "sigma_value") ** 2

        return X, Y, np.diag(W) if W is not None else None

    def _prepare_subset(self, subset):
        stations = self._dataset.stations

        idx_names = list(subset.index.names)
        subset = subset.reset_index()

        idx_names.insert(1, "stn_id")
        subset.insert(1, "stn_id", subset.stn_pk.map(stations.stn_id))

        if "stn_sh" in stations.columns:
            idx_names.insert(2, "stn_sh")
            subset.insert(2, "stn_sh", subset.stn_pk.map(stations.stn_sh))
        if "stn_h" in stations.columns:
            idx_names.insert(2, "stn_h")
            subset.insert(2, "stn_h", subset.stn_pk.map(stations.stn_h))

        return subset.to_dataframe(), idx_names

    def _fill_sigmas(self, sigmas, columns):
        for col in columns:
            sigma_col = f"s{col}"
            if sigma_col in sigmas:
                sigmas[col] = sigmas[sigma_col].fillna(self._default_sigmas[sigma_col])
                sigmas.drop(columns=sigma_col, inplace=True)
            else:
                sigmas[col] = self._default_sigmas[sigma_col]

    def _melt_subset(self, subset, idx_names, type):
        if type not in ["meas", "sigma"]:
            raise ValueError("Type must be either 'meas' or 'sigma'.")

        measurement_columns = self._dataset.measurements.measurements_columns

        subset = subset.melt(
            id_vars=idx_names,
            value_vars=measurement_columns,
            var_name="meas_type",
            value_name=f"{type}_value",
        )
        return subset

    def _merge_with_controls(self, data, coordinates):
        cols = ["stn", "trg"]
        for point in cols:
            data = data.merge(
                coordinates,
                how="left",
                left_on=f"{point}_id",
                right_on="id",
                suffixes=[f"_{col}" for col in cols],
            )
        return data

    def _calculate_coord_differences(self, data):
        for col in self._dataset.controls.coordinates_columns:
            if col != "z":
                data[f"d{col}"] = data[f"{col}_trg"] - data[f"{col}_stn"]
            else:
                data["dz"] = (
                    data["z_trg"] + data["trg_h"] - data[f"z_stn"] - data[f"stn_h"]
                )

    def _merge_orientations(self, data):
        data = data.merge(self._dataset.stations.orientation, how="left", on="stn_pk")
        data = data.merge(
            self._matrix_x_indexer.orientations_indices, how="left", on="stn_pk"
        )
        return data
