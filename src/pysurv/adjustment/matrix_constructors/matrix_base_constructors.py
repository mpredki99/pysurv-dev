# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

from abc import ABC, abstractmethod

import numpy as np

from pysurv.data.dataset import Dataset

from ..config_sigma import config_sigma
from .indexer_matrix_x import IndexerMatrixX
from .obs_equations_adapter import obs_eqations_adapter


class MatrixConstructor(ABC):
    """
    Abstract base class for constructing matrices for leas squares adjustment of
    surveying control network.

    This class provides the interface for building matrices from a given `Dataset`.
    Matrix_x_indexer is helper object that maps control point coordinate indices
    and station orinetation indices into matrix X columns.

    Subclasses should implement the `build` method to return the constructed matrix.
    """

    def __init__(self, dataset: Dataset, matrix_x_indexer: IndexerMatrixX) -> None:
        self._dataset = dataset
        self._matrix_x_indexer = matrix_x_indexer

    @abstractmethod
    def build(self):
        """Build and return the constructed matrix."""
        pass


class InnerConstraintsConstructor(MatrixConstructor):
    """
    Abstract base class for constructing inner contraints R matrix, list of that constraints
    and control point corrections matrix sX.

    This class initialize matrices constructor with specified number of columns in design matrix X.
    """

    def __init__(self, dataset, matrix_x_indexer, matrix_x_n_col: int) -> None:
        super().__init__(dataset, matrix_x_indexer)
        self._matrix_x_n_col = matrix_x_n_col


class MatrixXYWsWConstructor(MatrixConstructor):
    """
    Abstract base class for constructing X, Y, W, and sW matrices for adjustment computations.

    This class initialize matrices constructor with specified default sigmas row from sigma_config
    object.
    """

    def __init__(
        self,
        dataset: Dataset,
        matrix_x_indexer: IndexerMatrixX,
        default_sigmas_index: str,
    ) -> None:
        super().__init__(dataset, matrix_x_indexer)

        default_sigmas_index = default_sigmas_index or config_sigma.default_index
        self._default_sigmas = config_sigma[default_sigmas_index]


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
        matrix_x_indexer: IndexerMatrixX,
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
        matrix_x_row[matrix_x_output_indices] = coeficients
        matrix_y[matrix_y_row_idx] = free_term


class MatrixSWConstructor(MatrixXYWsWConstructor):
    """
    Abstract base class for constructing the controls weight matrix (sW) for adjustment computations.

    This class provides the interface and common functionality for building the diagonal controls weight
    matrix (sW).
    """

    def __init__(
        self,
        dataset: Dataset,
        matrix_x_indexer: IndexerMatrixX,
        default_sigmas_index: str | None,
    ) -> None:
        super().__init__(dataset, matrix_x_indexer, default_sigmas_index)

    @abstractmethod
    def build(self, matrix_x_n_col: int):
        """Build and return the diagonal controls weight matrix (sW)."""
        pass

    def _initialize_sw_matrix(self, matrix_x_n_col: int):
        """Initialize and return sW vector of length matrix_x_n_col."""
        return np.zeros(matrix_x_n_col)
