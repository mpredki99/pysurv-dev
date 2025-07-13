from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from .obs_equations_adapter import obs_eqations_adapter


class MatricesBuilder(ABC):

    def __init__(self, parent):
        self._parent = parent
        
    @property
    def dataset(self):
        return self._parent.dataset
    
    @property
    def cordinates_index_in_x_matrix(self):
        return self._parent.cordinates_index_in_x_matrix
    
    @property
    def orientations_index_in_x_matrix(self):
        return self._parent.orientations_index_in_x_matrix
    
    @property
    def default_sigmas(self):
        return self._parent.default_sigmas

    @abstractmethod
    def build_xyw(self, calculate_weights):
        pass
    
    @abstractmethod
    def build_sw(self):
        pass

    def initialize_xyw_matrices(self, calculate_weights):
        n_measurements = (
            self.dataset.measurements.measurement_values.count().sum()
        )
        n_coords = self.dataset.controls.coordinates.count().sum()
        n_orientations = (
            self.dataset.stations.orientation.count()
            if "orientation" in self.dataset.stations
            else 0
        )
        X = np.zeros((n_measurements, n_coords + n_orientations))
        Y = np.zeros((n_measurements, 1))
        W = np.zeros(n_measurements) if calculate_weights else None

        return X, Y, W
    
    def initialize_sw_matrix(self):
        return np.zeros(self._parent.matrix_X.shape[1])

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
