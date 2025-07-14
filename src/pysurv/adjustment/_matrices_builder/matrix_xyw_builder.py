from abc import abstractmethod

import numpy as np

from .matrix_xyw_sw_builder import MatrixXYWsWBuilder
from .obs_equations_adapter import obs_eqations_adapter


class MatrixXYWBuilder(MatrixXYWsWBuilder):
    def __init__(self, dataset, matrix_x_indexer, default_sigmas):
        super().__init__(dataset, matrix_x_indexer, default_sigmas)

    @abstractmethod
    def build(self, calculate_weights):
        pass

    def _initialize_xyw_matrices(self, calculate_weights):
        n_measurements = self._dataset.measurements.measurement_values.count().sum()
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
        measurement_type,
        value,
        coord_diff,
        matrix_x_col_indices,
        matrix_x_row,
        matrix_y_row_idx,
        matrix_y_row,
    ):
        observation_func = obs_eqations_adapter[measurement_type]
        matrix_x_output_indices, coeficients, free_term = observation_func(
            value, coord_diff, matrix_x_col_indices
        )
        matrix_x_row[[*matrix_x_output_indices]] = coeficients
        matrix_y_row[matrix_y_row_idx] = free_term
