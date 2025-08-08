# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

from functools import cached_property
from warnings import warn

import numpy as np
from numpy.linalg import pinv

from pysurv.warnings import SVDNotConvergeWarning

from ._constants import INVALID_INDEX
from .adjustment_iteration import AdjustmentIteration
from .matrices import Matrices


class DenseIteration(AdjustmentIteration):
    """Class that implements LSQ adjustment iteration on numpy dense format stored matrices."""

    def __init__(self, matrices: Matrices) -> None:
        super().__init__(matrices)

        coordinate_indices = self._lsq_matrices.indexer.coordinate_indices.values
        self._coord_mask = self._get_coord_mask(coordinate_indices)
        self._coord_idx = self._get_coord_idx(coordinate_indices)

        self._matrix_g = None
        self._inv_matrix_G = None
        self._cross_product = None
        self._increments = None

    @property
    def matrix_G(self):
        """Return matrix G."""
        return self._matrix_g

    @property
    def inv_g_matrix(self):
        """Return inverse of G matrix."""
        return self._inv_matrix_G

    @property
    def cross_product(self):
        """Rerurn cross product"""
        return self._cross_product

    @property
    def increments(self):
        """Return increments."""
        return self._increments

    @cached_property
    def coord_increments(self):
        """Return fitered for just coordinate increments."""
        return self.increments[self._coord_idx]

    @cached_property
    def increment_matrix(self):
        """Return increment matrix."""
        return self._get_increment_matrix()

    @cached_property
    def obs_residuals(self):
        """Return observation residuals."""
        X = self._lsq_matrices.matrix_X
        Y = self._lsq_matrices.matrix_Y
        return X @ self.increments - Y

    @cached_property
    def residual_variance(self):
        """Return residual variance."""
        return self._get_residual_variance()

    @cached_property
    def covariance_X(self):
        """Return the covariance matrix of X."""
        return self.residual_variance * self.inv_g_matrix

    @cached_property
    def covariance_Y(self):
        """Return the covariance matrix of Y."""
        X = self._lsq_matrices.matrix_X
        return X @ self.covariance_X @ X.T

    @cached_property
    def covariance_r(self):
        """Return the covariance matrix of residuals."""
        return self._get_cov_r()

    @cached_property
    def point_weights(self):
        """Return the point weights."""
        return self._get_point_weights()

    def _get_coord_mask(self, coord_indices):
        """Return a boolean mask for valid coordinate indices."""
        return coord_indices != INVALID_INDEX

    def _get_coord_idx(self, coord_indices):
        """Return the indices of valid coordinates."""
        return coord_indices[self._coord_mask]

    def _get_matrix_g(self):
        """Calculate the gram matrix."""
        X = self._lsq_matrices.matrix_X
        W = self._lsq_matrices.matrix_W
        R = self._lsq_matrices.matrix_R
        sX = self._lsq_matrices.matrix_sX
        sW = self._lsq_matrices.matrix_sW

        if W is not None:
            G1 = X.T @ W @ X
        else:
            G1 = X.T @ X

        if sX is not None and sW is not None:
            G2 = sX.T @ sW @ sX
        elif R is not None and sW is not None:
            G2 = R.T @ R @ sW
        else:
            G2 = None

        return G1 + G2 if G2 is not None else G1

    def _get_cross_product(self):
        """Calculate the cross product."""
        X = self._lsq_matrices.matrix_X
        Y = self._lsq_matrices.matrix_Y
        W = self._lsq_matrices.matrix_W

        if W is not None:
            return X.T @ W @ Y
        else:
            return X.T @ Y

    def _get_increment_matrix(self):
        """Calculate increment matrix."""
        increments_matrix = np.zeros(self._coord_mask.shape)
        increments_matrix[self._coord_mask] += self.coord_increments.flatten()
        return increments_matrix

    def _get_residual_variance(self):
        """Calculate the residual variance."""
        W = self._lsq_matrices.matrix_W
        k = self._lsq_matrices.degrees_of_freedom
        residuals = self.obs_residuals.reshape(-1)

        if W is not None and k > 0:
            return np.divide((residuals**2 * W.diagonal()).sum(), k)
        elif W is None and k > 0:
            return np.divide((residuals**2).sum(), k)
        else:
            return 1

    def _get_cov_r(self):
        """Calculate covariance matrix of residuals."""
        n = self.covariance_Y.shape[0]
        W = self._lsq_matrices.matrix_W

        diag_idx = np.diag_indices(n)
        weight_coffactor_matrix = pinv(W) if W is not None else np.eye(n)
        weight_coffactor_matrix[diag_idx] *= self.residual_variance

        return weight_coffactor_matrix - self.covariance_Y

    def _get_point_weights(self):
        """Calculate the point weights from sW matrix."""
        sW = self._lsq_matrices.matrix_sW
        if sW is not None:
            return sW.diagonal()[self._coord_idx]
        return

    def _reset_cache(self):
        """Reset cached properties values."""
        for name in dir(self.__class__):
            attr = getattr(self.__class__, name)
            if isinstance(attr, cached_property) and name in self.__dict__:
                del self.__dict__[name]

    def run(self):
        """Run a single iteration step."""
        try:
            matrix_g = self._get_matrix_g()
            inv_matrix_G = pinv(matrix_g)
            cross_product = self._get_cross_product()
            increments = inv_matrix_G @ cross_product

            self._reset_cache()

            self._counter += 1
            self._matrix_g = matrix_g
            self._inv_matrix_G = inv_matrix_G
            self._cross_product = cross_product
            self._increments = increments

            return True

        except np.linalg.LinAlgError:
            warn(
                f"Calculations aborted due to SVD did not converge in {self._counter + 1}. iteration.",
                SVDNotConvergeWarning,
            )
            return False
