# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

from abc import ABC, abstractmethod

from .adjustment_matrices import AdjustmentMatrices


class AdjustmentIteration(ABC):
    """Abstract base class for LSQ iteration strategy objects."""

    def __init__(self, matrices: AdjustmentMatrices) -> None:
        self._lsq_matrices = matrices
        self._counter = 0

    def __bool__(self):
        return self._counter > 0

    @property
    def counter(self):
        """Return the current iteration counter."""
        return self._counter

    @property
    @abstractmethod
    def matrix_G(self):
        """Return matrix G."""
        pass

    @property
    @abstractmethod
    def inv_matrix_G(self):
        """Return inverse of G matrix."""
        pass

    @property
    @abstractmethod
    def cross_product(self):
        """Rerurn cross product"""
        pass

    @property
    @abstractmethod
    def increments(self):
        """Return increments."""
        pass

    @property
    @abstractmethod
    def coord_increments(self):
        """Return fitered for just coordinate increments."""
        pass

    @property
    @abstractmethod
    def increment_matrix(self):
        """Return increment matrix."""
        pass

    @property
    @abstractmethod
    def obs_residuals(self):
        """Return observation residuals."""
        pass

    @property
    @abstractmethod
    def residual_variance(self):
        """Return residual variance."""
        pass

    @property
    @abstractmethod
    def covariance_X(self):
        """Return the covariance matrix of X."""
        pass

    @property
    @abstractmethod
    def covariance_Y(self):
        """Return the covariance matrix of Y."""
        pass

    @property
    @abstractmethod
    def covariance_r(self):
        """Return the covariance matrix of residuals."""
        pass

    @property
    @abstractmethod
    def coordinate_weights(self):
        """Return the point weights."""
        pass

    @abstractmethod
    def run(self):
        """Run the LSQ iteration."""
        pass
