# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

from typing import Tuple

import numpy as np
import pandas as pd

from pysurv.data.dataset import Dataset

from .indexer_matrix_x import IndexerMatrixX
from .matrix_base_constructors import MatrixXYWConstructor


class SpeedXYWConstructor(MatrixXYWConstructor):
    """
    SpeedXYWConstructor constructs the design matrix (X), observation vector (Y), and weight matrix (W)
    for least squares adjustment computations using a fast, vectorized approach.
    """

    def __init__(
        self,
        dataset: Dataset,
        matrix_x_indexer: IndexerMatrixX,
        default_sigmas_index: str | None,
    ) -> None:
        super().__init__(dataset, matrix_x_indexer, default_sigmas_index)
        self._prepared_dataset = None

    def _prepare_dataset(self, calculate_weights) -> None:
        index_names = self._prepare_measurement_data()
        if calculate_weights:
            self._prepare_sigma_data(index_names)

    def _prepare_measurement_data(self) -> list[str]:
        measurement_data = self._dataset.measurements.measurement_data
        station_columns = self._dataset.stations.columns

        index_names = list(measurement_data.index.names)
        index_names.extend(station_columns)

        measurement_data = self._merge_stations(measurement_data)
        measurement_data = self._melt_subset(measurement_data, index_names, "meas")

        self._prepared_dataset = measurement_data.dropna(subset="meas_value")
        self._merge_coordinte_indices()
        if "orientation" in station_columns:
            self._merge_orientation_indices()
        return index_names

    def _prepare_sigma_data(self, index_names) -> None:
        sigma_data = self._dataset.measurements.sigma_data

        sigma_data = self._merge_stations(sigma_data)
        sigma_data = self._fill_with_default_sigmas(sigma_data)
        sigma_data = self._melt_subset(sigma_data, index_names, "sigma")

        merge_on_cols = index_names.append("meas_type")
        self._prepared_dataset = self._prepared_dataset.merge(
            sigma_data, how="left", on=merge_on_cols
        )

    def build(
        self, calculate_weights: bool
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Builds and returns the X, Y, and W matrices for least squares adjustment."""
        X, Y, W = self._initialize_xyw_matrices(calculate_weights)

        if self._prepared_dataset is None:
            self._prepare_dataset(calculate_weights)
        elif "orientation" in self._prepared_dataset.columns:
            self._update_orientations()
        self._coord_differences()

        for eq_row, row in enumerate(self._prepared_dataset.itertuples(index=False)):
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
                W[eq_row] = self._get_weight(getattr(row, "sigma_value"))

        return X, Y, np.diag(W) if W is not None else None

    def _get_weight(self, sigma_value: float) -> float:
        if sigma_value > 0:
            return 1 / sigma_value**2
        else:
            return 0

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

    def _merge_coordinte_indices(self) -> None:
        coordinate_indices = self._matrix_x_indexer.coordinate_indices

        end_points = ["stn", "trg"]
        for point in end_points:
            self._prepared_dataset = self._prepared_dataset.merge(
                coordinate_indices,
                how="left",
                left_on=f"{point}_id",
                right_on="id",
                suffixes=[f"_{point}" for point in end_points],
            )

    def _merge_orientation_indices(self) -> None:
        orientation_indices = self._matrix_x_indexer.orientation_indices
        self._prepared_dataset = self._prepared_dataset.merge(
            orientation_indices, how="left", on="stn_pk"
        )
        return self._prepared_dataset

    def _update_orientations(self) -> pd.DataFrame:
        orientations = self._dataset.stations.orientation
        self._prepared_dataset["orientation"] = self._prepared_dataset.stn_pk.map(
            orientations
        )

    def _coord_differences(self) -> None:
        coordinates = self._dataset.controls.coordinates

        for coord_col in coordinates.columns:
            stn_coords = self._prepared_dataset.stn_id.map(coordinates[coord_col])
            trg_coords = self._prepared_dataset.trg_id.map(coordinates[coord_col])

            if coord_col != "z":
                coord_diff = f"d{coord_col}"
                self._prepared_dataset[coord_diff] = trg_coords - stn_coords

            else:
                self._prepared_dataset["dz"] = (
                    trg_coords
                    + self._prepared_dataset["trg_h"]
                    - stn_coords
                    - self._prepared_dataset["stn_h"]
                )
