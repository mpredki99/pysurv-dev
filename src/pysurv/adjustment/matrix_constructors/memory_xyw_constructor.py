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


class MemoryXYWConstructor(MatrixXYWConstructor):
    """
    Constructs the design matrix (X), observation vector (Y), and weight matrix (W)
    for least squares adjustment computations using saving memory, row-wise approach.

    This class iterates over the measurement data and builds the matrices required for
    adjustment computations, reading data from origin objects.
    """

    def __init__(
        self,
        dataset: Dataset,
        matrix_x_indexer: IndexerMatrixX,
        default_sigmas: pd.Series,
    ) -> None:
        super().__init__(dataset, matrix_x_indexer, default_sigmas)

    def build(
        self, calculate_weights: bool
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Builds and returns the X, Y, and W matrices for least squares adjustment using memory-efficient, row-wise approach."""
        measurements = self._dataset.measurements
        controls = self._dataset.controls
        stations = self._dataset.stations

        coordinate_indices = self._matrix_x_indexer.coordinate_indices
        orientation_index = self._matrix_x_indexer.orientation_indices

        X, Y, W = self._initialize_xyw_matrices(calculate_weights)

        stn_pk_loc = measurements.index.names.index("stn_pk")
        trg_id_loc = measurements.index.names.index("trg_id")
        trg_h_loc = (
            measurements.index.names.index("trg_h")
            if "trg_h" in measurements.index.names
            else None
        )

        eq_row = 0
        for idx in measurements.index:
            stn_pk = idx[stn_pk_loc]
            stn_id = stations.at[stn_pk, "stn_id"]
            stn_h = stations.at[stn_pk, "stn_h"] if "stn_h" in stations.columns else 0

            trg_id = idx[trg_id_loc]
            trg_h = idx[trg_h_loc] if trg_h_loc else 0

            coord_diff = {
                f"d{col}": controls.at[trg_id, col] - controls.at[stn_id, col]
                for col in controls.coordinate_columns
                if col != "z"
            }
            if "z" in controls.columns:
                coord_diff["dz"] = (
                    controls.at[trg_id, "z"] + trg_h - controls.at[stn_id, "z"] - stn_h
                )

            for meas_type in measurements.measurement_columns:
                meas_value = measurements.at[idx, meas_type]
                if np.isnan(meas_value):
                    continue

                matrix_x_col_index = {
                    f"{coord}_{label}": coordinate_indices.at[ctrl_id, coord]
                    for ctrl_id, label in zip([stn_id, trg_id], ["stn", "trg"])
                    for coord in controls.coordinate_columns
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

    def _get_sigma_value(self, idx: pd.Index, meas_type: str) -> float:
        """Return the sigma value for a given measurement index and type."""
        sigma_type = f"s{meas_type}"
        sigma_value = None

        if sigma_type in self._dataset.measurements.sigma_columns:
            sigma_value = self._dataset.measurements.at[idx, sigma_type]
        if pd.isna(sigma_value):
            return self._default_sigmas[sigma_type]
        return sigma_value
