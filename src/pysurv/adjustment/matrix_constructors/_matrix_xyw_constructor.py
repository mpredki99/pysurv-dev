# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

from abc import abstractmethod

import numpy as np

from pysurv.data.dataset import Dataset

from ._matrix_x_indexer import MatrixXIndexer
from ._matrix_xyw_sw_constructor import MatrixXYWsWConstructor
from ._obs_equations_adapter import obs_eqations_adapter


class MatrixXYWConstructor(MatrixXYWsWConstructor):
    """
    Abstract base class for constructing X, Y, and W matrices for adjustment computations.

    This class provides the interface and common functionality for building the design matrix (X),
    observation vector (Y), and weight matrix (W) used in least squares adjustment computations.
    Subclasses should implement the `build` method to construct these matrices.
    """

    def __init__(
        self,
        dataset: Dataset,
        matrix_x_indexer: MatrixXIndexer,
        default_sigmas_index: str | None,
    ) -> None:
        super().__init__(dataset, matrix_x_indexer, default_sigmas_index)

    @abstractmethod
    def build(self, calculate_weights: bool):
        """Build and return matrix X, vector Y, and optionally weight matrix W."""
        pass

    def _initialize_xyw_matrices(self, calculate_weights: bool):
        """Initialize and return empty X, Y, and optionally W matrix."""
        n_measurements = self._dataset.measurements.measurement_data.count().sum()
        n_coords = self._dataset.controls.coordinates.count().sum()
        n_orientations = (
            self._dataset.stations.orientation.count()
            if "orientation" in self._dataset.stations
            else 0
        )
        X = np.zeros((n_measurements, n_coords + n_orientations))
        Y = np.zeros((n_measurements, 1))
        W = np.zeros(n_measurements) if calculate_weights else None

        return X, Y, W

    def _apply_observation_function(
        self,
        measurement_type: str,
        value: float,
        coord_diff: dict,
        matrix_x_col_indices: dict,
        matrix_x_row: np.ndarray,
        matrix_y_row_idx: int,
        matrix_y: np.ndarray,
    ):
        """Apply the observation function for a given measurement type to update X and Y."""
        observation_func = obs_eqations_adapter[measurement_type]
        matrix_x_output_indices, coeficients, free_term = observation_func(
            value, coord_diff, matrix_x_col_indices
        )
        matrix_x_row[[*matrix_x_output_indices]] = coeficients
        matrix_y[matrix_y_row_idx] = free_term
