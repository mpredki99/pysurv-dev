# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

import numpy as np
from numpy.linalg import pinv

from ._constants import INVALID_INDEX
from .iteration import Iteration
from .matrices import Matrices


class IterationDense(Iteration):
    """Class that implements LSQ adjustment on numpy dense format stored matrices."""

    def __init__(self, lsq_matrices: Matrices) -> None:
        super().__init__(lsq_matrices)
        self._counter = 0
        self._N_inv = None
        self._L = None
        self._increments = None
        self._coord_increments = None
        self._increments_matrix = None
        self._point_weights = None
        self._obs_residuals = None
        self._residual_variance = None
        self._cov_X = None
        self._cov_Y = None
        self._cov_r = None

        coordinate_indices = lsq_matrices.indexer.coordinate_indices.values
        self._coord_mask = self._get_coord_mask(coordinate_indices)
        self._coord_idx = self._get_coord_idx(coordinate_indices)

    @property
    def counter(self):
        """Return the current iteration counter."""
        return self._counter

    @property
    def inv_gram_matrix(self):
        """Return the inverse Gram matrix."""
        return self._N_inv

    @property
    def cross_product(self):
        """Return the cross product vector."""
        return self._L

    @property
    def increments(self):
        """Return the increments vector."""
        return self._increments

    @property
    def increments_matrix(self):
        """Return the increments matrix."""
        return self._increments_matrix

    @property
    def obs_residuals(self):
        """Return the observation residuals."""
        return self._obs_residuals

    @property
    def residual_variance(self):
        """Return the residual variance."""
        return self._residual_variance

    @property
    def covariance_X(self):
        """Return the covariance matrix of X."""
        return self._cov_X

    @property
    def covariance_Y(self):
        """Return the covariance matrix of Y."""
        return self._cov_Y

    @property
    def covariance_r(self):
        """Return the covariance matrix of residuals."""
        return self._cov_r

    @property
    def point_weights(self):
        """Return the point weights."""
        return self._point_weights

    def _get_coord_mask(self, coord_indices):
        """Return a boolean mask for valid coordinate indices."""
        return coord_indices != INVALID_INDEX

    def _get_coord_idx(self, coord_indices):
        """Return the indices of valid coordinates."""
        return coord_indices[self._coord_mask]

    def _calculate_normal_equations(self):
        """Calculate the normal equations for the adjustment."""
        X = self._lsq_matrices.matrix_X
        Y = self._lsq_matrices.matrix_Y
        W = self._lsq_matrices.matrix_W
        R = self._lsq_matrices.matrix_R
        sX = self._lsq_matrices.matrix_sX
        sW = self._lsq_matrices.matrix_sW

        if W is not None:
            N1 = X.T @ W @ X
            self._L = X.T @ W @ Y
        else:
            N1 = X.T @ X
            self._L = X.T @ Y

        if sX is not None and sW is not None:
            N2 = sX.T @ sW @ sX
        elif R is not None and sW is not None:
            N2 = R.T @ R @ sW
        else:
            N2 = None

        N = N1 + N2 if N2 is not None else N1

        self._N_inv = pinv(N)

    def _calculate_increment_matrix(self):
        """Calculate the increments matrix for coordinates."""
        self._increments_matrix = np.zeros(self._coord_mask.shape)
        self._calculate_coord_increments()
        self._increments_matrix[self._coord_mask] += self._coord_increments

    def _calculate_coord_increments(self):
        """Calculate the increments for coordinates only."""
        self._calculate_increments()
        increments = self.increments.flatten()
        self._coord_increments = increments[self._coord_idx]

    def _calculate_increments(self):
        """Calculate the increments vector."""
        self._increments = self._N_inv @ self._L

    def _calculate_point_weights(self):
        """Calculate the point weights from sW matrix."""
        sW = self._lsq_matrices.matrix_sW
        if sW is not None:
            self._point_weights = self._lsq_matrices.matrix_sW.diagonal()
            self._point_weights = self._point_weights[self._coord_idx]

    def _calculate_obs_residuals(self):
        """Calculate the observation residuals."""
        X = self._lsq_matrices.matrix_X
        Y = self._lsq_matrices.matrix_Y
        self._obs_residuals = X @ self.increments - Y

    def _calculate_residual_variance(self):
        """Calculate the residual variance."""
        W = self._lsq_matrices.matrix_W
        k = self._lsq_matrices.degrees_of_freedom
        residuals = self.obs_residuals.reshape(-1)

        if W is not None and k > 0:
            self._residual_variance = np.divide((residuals**2 * W.diagonal()).sum(), k)
        elif W is None and k > 0:
            self._residual_variance = np.divide((residuals**2).sum(), k)
        else:
            self._residual_variance = 1

    def _calculate_covariance_matrices(self):
        """Calculate the covariance matrices for X, Y, and residuals."""
        residual_variance = self.residual_variance
        X = self._lsq_matrices.matrix_X
        W = self._lsq_matrices.matrix_W

        self._cov_X = residual_variance * self.inv_gram_matrix
        self._cov_Y = X @ self._cov_X @ X.T

        n = self._cov_Y.shape[0]

        diag_idx = np.diag_indices(n)
        weight_coffactor_matrix = pinv(W) if W is not None else np.eye(n)
        weight_coffactor_matrix[diag_idx] *= residual_variance

        self._cov_r = weight_coffactor_matrix - self._cov_Y

    def run(self):
        """Run a single dense iteration step."""
        self._counter += 1
        self._calculate_normal_equations()
        self._calculate_increment_matrix()
        self._calculate_point_weights()
        self._calculate_obs_residuals()
        self._calculate_residual_variance()
        self._calculate_covariance_matrices()
