from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from .obs_equations_adapter import obs_eqations_adapter


class LSQMatrixBuildStrategy(ABC):

    def __init__(self, parent):
        self._parent = parent
        self.calculate_weights = self._parent.method != "ordinary"

    @abstractmethod
    def build(self):
        pass

    def _initialize_xyw_matrices(self):
        n_measurements = (
            self._parent.dataset.measurements.measurement_values.count().sum()
        )
        n_coords = self._parent.dataset.controls.coordinates.count().sum()
        n_orientations = (
            self._parent.dataset.stations.orientation.count()
            if "orientation" in self._parent.dataset.stations
            else 0
        )
        X = np.zeros((n_measurements, n_coords + n_orientations))
        Y = np.zeros((n_measurements, 1))
        W = np.zeros(n_measurements) if self.calculate_weights else None

        return X, Y, W

    def apply_observation_function(
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
