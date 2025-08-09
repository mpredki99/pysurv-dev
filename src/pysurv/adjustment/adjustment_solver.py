# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

from abc import ABC, abstractmethod

from pysurv.data.controls import Controls

from .adjustment_matrices import AdjustmentMatrices
from .config_solver import config_solver
from .dense_iteration import DenseIteration


class AdjustmentSolver(ABC):
    def __init__(
        self,
        controls: Controls,
        lsq_matrices: AdjustmentMatrices,
        config_solver_index: str | None = None,
        create_list_of_variances: bool = False,
    ) -> None:
        self._controls = controls
        self._matrices = lsq_matrices
        self._approx_coordinates = self._controls.coordinates.copy()
        self._create_list_of_variances = create_list_of_variances
        self._residual_variances = []
        self._coord_corrections_variances = []
        self._n_movable_tie_points = self._get_n_movable_tie_points()
        self._config_solver = self._get_config_solver(config_solver_index)
        self._results = None

        self._iteration = self._get_lsq_iteration()
        self._matrices.methods._inject_solver(self)

    @property
    def matrices(self):
        """Return LSQ matrices."""
        return self._matrices

    @property
    def methods(self):
        return self._matrices.methods

    @property
    def results(self):
        """Prepare if needed and return adjustment results."""
        if self._results is None and self._iteration:
            self._prepare_adjustment_results()
        return self._results

    @property
    @abstractmethod
    def n_iter(self):
        pass

    @property
    @abstractmethod
    def matrix_G(self):
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

    @property
    @abstractmethod
    def n_movable_tie_points(self):
        pass

    @property
    @abstractmethod
    def coord_cor_variance(self):
        pass

    @property
    @abstractmethod
    def coord_corrections(self):
        pass

    @property
    @abstractmethod
    def svd_converge(self):
        pass

    @abstractmethod
    def solve(self):
        """Run the adjustment process."""
        pass

    @abstractmethod
    def iterate(self):
        """Perform a single iteration of the adjustment."""
        pass

    @abstractmethod
    def update_matrices(self):
        """Update X, Y, and weight matrices."""
        pass

    @abstractmethod
    def _prepare_adjustment_results(self):
        pass

    def _get_config_solver(self, index: str | None):
        """Returns config_solver row."""
        if index is None:
            index = config_solver.default_index
        return config_solver[index]

    def _get_n_movable_tie_points(self):
        """Get number of movable reference points."""
        if self._matrices.matrix_sW is None:
            return self._count_movable_tie_points_from_indexer()
        return self._count_movable_tie_points_from_sw()

    def _count_movable_tie_points_from_indexer(self):
        """
        Count how many tie points are movable based on indexer object. If sW matrix is None
        (ordinary free adjustment), than all control points are movable tie points.
        """
        return self._matrices.indexer.coordinate_indices.max().max() + 1

    def _count_movable_tie_points_from_sw(self):
        """
        Count how many tie points are movable based on control point weight matrix.
        If sW matrix is not None, than tie control points have non-zero weights.
        Control points with zero weights are not tie points.
        """
        sW = self._matrices.matrix_sW
        return sW.diagonal()[sW.diagonal() > 0].size

    def _get_lsq_iteration(self):
        """Returns iteration object."""
        return DenseIteration(self._matrices)
