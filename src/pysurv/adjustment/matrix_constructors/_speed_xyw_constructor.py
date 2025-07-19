# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

from typing import Tuple

import numpy as np
import pandas as pd

from pysurv.data.dataset import Dataset

from ._matrix_x_indexer import MatrixXIndexer
from ._matrix_xyw_constructor import MatrixXYWConstructor


class SpeedXYWConstructor(MatrixXYWConstructor):
    """
    SpeedXYWConstructor constructs the design matrix (X), observation vector (Y), and weight matrix (W)
    for least squares adjustment computations using a fast, vectorized approach.
    """

    def __init__(
        self,
        dataset: Dataset,
        matrix_x_indexer: MatrixXIndexer,
        default_sigmas_index: str | None,
    ) -> None:
        super().__init__(dataset, matrix_x_indexer, default_sigmas_index)

    def build(
        self, calculate_weights: bool
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Builds and returns the X, Y, and W matrices for least squares adjustment."""
        measurement_data = self._dataset.measurements.measurement_data
        sigma_data = self._dataset.measurements.sigma_data
        coordinates = self._dataset.controls.coordinates

        station_columns = self._dataset.stations.columns
        coordinate_indices = self._matrix_x_indexer.coordinate_indices

        X, Y, W = self._initialize_xyw_matrices(calculate_weights)

        index_names = list(measurement_data.index.names)
        index_names.extend(station_columns)

        measurement_data = self._merge_stations(measurement_data)
        measurement_data = self._melt_subset(measurement_data, index_names, "meas")
        measurement_data.dropna(subset="meas_value", inplace=True)

        if calculate_weights:
            sigma_data = self._merge_stations(sigma_data)
            sigma_data = self._fill_with_default_sigmas(sigma_data)
            sigma_data = self._melt_subset(sigma_data, index_names, "sigma")

            merge_on_cols = index_names.append("meas_type")
            measurement_data = measurement_data.merge(
                sigma_data, how="left", on=merge_on_cols
            )

        measurement_data = self._merge_controls(measurement_data, coordinates)
        measurement_data = self._coord_differences(measurement_data)
        measurement_data = self._merge_controls(measurement_data, coordinate_indices)

        if "orientation" in station_columns:
            measurement_data = self._merge_orientations(measurement_data)

        for eq_row, row in enumerate(measurement_data.itertuples(index=False)):
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

    def _merge_stations(self, subset: pd.DataFrame) -> pd.DataFrame:
        """Merge station attributes into the subset."""
        stations = self._dataset.stations
        subset = subset.reset_index()
        subset = subset.merge(stations, how="left", on="stn_pk")

        return subset

    def _fill_with_default_sigmas(self, sigma_data: pd.DataFrame) -> pd.DataFrame:
        """Fill missing sigma values in sigma_data with default sigma values."""
        measurement_columns = self._dataset.measurements.measurement_columns
        for meas_col in measurement_columns:
            sigma_col = f"s{meas_col}"
            if sigma_col in sigma_data:
                sigma_data[meas_col] = sigma_data[sigma_col].fillna(
                    self._default_sigmas[sigma_col]
                )
                sigma_data.drop(columns=sigma_col, inplace=True)
            else:
                sigma_data[meas_col] = self._default_sigmas[sigma_col]
        return sigma_data

    def _melt_subset(
        self, subset: pd.DataFrame, index_names: list, type: str
    ) -> pd.DataFrame:
        """Reshape subset DataFrame from wide to long format for measurements or sigmas."""
        if type not in ["meas", "sigma"]:
            raise ValueError("Type must be either 'meas' or 'sigma'.")

        measurement_columns = self._dataset.measurements.measurement_columns
        subset = subset.melt(
            id_vars=index_names,
            value_vars=measurement_columns,
            var_name="meas_type",
            value_name=f"{type}_value",
        )
        return subset

    def _merge_controls(
        self, measurement_data: pd.DataFrame, ctrl_data: pd.DataFrame
    ) -> pd.DataFrame:
        end_points = ["stn", "trg"]
        for point in end_points:
            measurement_data = measurement_data.merge(
                ctrl_data,
                how="left",
                left_on=f"{point}_id",
                right_on="id",
                suffixes=[f"_{point}" for point in end_points],
            )
        return measurement_data

    def _coord_differences(self, measurement_data: pd.DataFrame) -> pd.DataFrame:
        coord_columns = self._dataset.controls.coordinate_columns
        for coord_col in coord_columns:
            if coord_col != "z":
                coord_stn = f"{coord_col}_stn"
                coord_trg = f"{coord_col}_trg"
                coord_diff = f"d{coord_col}"
                measurement_data[coord_diff] = (
                    measurement_data[coord_trg] - measurement_data[coord_stn]
                )
                measurement_data.drop(columns=coord_stn, inplace=True)
                measurement_data.drop(columns=coord_trg, inplace=True)
            else:
                measurement_data["dz"] = (
                    measurement_data["z_trg"]
                    + measurement_data["trg_h"]
                    - measurement_data["z_stn"]
                    - measurement_data["stn_h"]
                )
                measurement_data.drop(columns="z_stn", inplace=True)
                measurement_data.drop(columns="z_trg", inplace=True)
        return measurement_data

    def _merge_orientations(self, measurement_data: pd.DataFrame) -> pd.DataFrame:
        orientation_indices = self._matrix_x_indexer.orientation_indices
        measurement_data = measurement_data.merge(
            orientation_indices, how="left", on="stn_pk"
        )
        return measurement_data
