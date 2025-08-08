# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

import numpy as np

from . import robust
from .matrices import Matrices
from .matrix_constructors.matrix_r_constructor import MatrixRConstructor
from .matrix_constructors.matrix_sx_constructor import MatrixSXConstructor


class DenseMatrices(Matrices):
    """
    Concrete implementation of the Matrices class for least squares adjustment.

    This class constructs and manages dense (NumPy ndarray-based) matrices required for least squares adjustment.
    """

    def _build_xyw_matrices(self) -> None:
        """Build the X, Y, and W matrices for least squares adjustment."""
        self._X, self._Y, self._W = self._xyw_sw_init_strategy.xyw_constructor.build(
            calculate_weights=self.calculate_weights
        )

    def _build_inner_constraints_matrix(self) -> None:
        """Build the R matrix and inner constraints list."""
        matrix_r_constructor = MatrixRConstructor(
            self._dataset, self.indexer, self.matrix_X.shape[1]
        )
        self._R, self._inner_constraints = matrix_r_constructor.build()

    def _build_sx_matrix(self):
        """Build the control point corrections matrix (sX)."""
        matrix_sx_constructor = MatrixSXConstructor(
            self._dataset, self.indexer, self.matrix_X.shape[1]
        )
        self._sX = matrix_sx_constructor.build()

    def _build_sw_matrix(self):
        """Build the control point weights matrix (sW)."""
        self._sW = self._xyw_sw_init_strategy.sw_constructor.build(
            self.matrix_X.shape[1]
        )

    def _is_obs_method_robust(self):
        """Check wether the observation weight matrix reweigt method is robust."""
        return self._methods.obs_tuning_constants is not None

    def _is_free_adj_method_robust(self):
        """Check wether the control point weights matrix reweigt method is robust."""
        return self._methods.free_adj_tuning_constants is not None

    def _update_weights(
        self, matrix: np.ndarray, v: np.ndarray, func: callable, tuning_constants: dict
    ) -> None:
        """Update proper weights matrix."""
        diag_idx = np.diag_indices(matrix.shape[0])
        matrix[diag_idx] *= func(v, **tuning_constants)

    def update_xy_matrices(self) -> None:
        """Update the X and Y matrices for least squares adjustment."""
        if "hz" in self._dataset.measurements.angular_measurement_columns:
            self._update_stations_orientation()
        self._X, self._Y, _ = self._xyw_sw_init_strategy.xyw_constructor.build(
            calculate_weights=False
        )

    def update_w_matrix(self, v: np.ndarray) -> None:
        """Update observation weights matrix."""
        if not self._is_obs_method_robust():
            return

        func = getattr(robust, self._methods.observations)
        self._update_weights(self._W, v, func, self._methods.obs_tuning_constants)

    def update_sw_matrix(self, v: np.ndarray) -> None:
        """Update control point weights matrix."""
        if not self._is_free_adj_method_robust():
            return

        func = getattr(robust, self._methods.free_adjustment)
        self._update_weights(self._sW, v, func, self._methods.free_adj_tuning_constants)
