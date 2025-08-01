# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

import numpy as np

from pysurv.data.controls import Controls

from .config_solver import config_solver
from .iteration_dense import IterationDense
from .matrices import Matrices


class Solver:
    """Class for solving surveying adjustment task."""

    def __init__(
        self,
        controls: Controls,
        lsq_matrices: Matrices,
        config_solver_index: str | None = None,
    ) -> None:
        self._controls = controls
        self._lsq_matrices = lsq_matrices
        self._solver_config = self._get_solver_config(config_solver_index)
        self._iteration = self._get_lsq_iteration()
        self._approx_coordinates = self._controls.copy()
        self._residual_variances = []
        self._coord_corrections = None
        self._coord_corrections_variances = []
        self._results = None

    def _get_lsq_iteration(self):
        return IterationDense(self._lsq_matrices)

    def _get_solver_config(self, index):
        if index is None:
            index = config_solver.default_index
        return config_solver[index]

    @property
    def lsq_matrices(self):
        """Return LSQ matrices."""
        return self._lsq_matrices

    @property
    def iteration(self):
        """Return current iteration object."""
        return self._iteration

    @property
    def results(self):
        """Prepare if needed and return adjustment results."""
        if self._results is None:
            self._prepare_adjustment_results()
        return self._results

    def _check_condition(self):
        """Check if iteration should stop or continue."""
        if self._solver_config.max_iter <= self._iteration._counter or all(
            self._iteration._coord_increments <= self._solver_config.threshold
        ):
            return self.results
        return self.solve()

    def _update_coord_corrections(self):
        """Update coordinate corrections for this iteration."""
        if self._iteration._counter == 1:
            self._coord_corrections = self._iteration._coord_increments
        else:
            self._coord_corrections += self._iteration._coord_increments

    def _append_residual_variances(self):
        """Append current residual and coordinate correction variances."""
        self._residual_variances.append(self._iteration.residual_variance)
        self._coord_corrections_variances.append(
            self._calculate_coord_corrections_variance()
        )

    def _calculate_coord_corrections_variance(self):
        """Calculate variance of coordinate corrections."""
        point_weights = self.iteration._point_weights
        if point_weights is not None:
            return np.divide(
                (self._coord_corrections**2 * point_weights).sum(),
                self._coord_corrections.shape[0],
            )
        else:
            return np.divide(
                (self._coord_corrections**2).sum(), self._coord_corrections.shape[0]
            )

    def _normalize_v(self, v: np.ndarray, sv: np.ndarray) -> np.ndarray:
        """Return normalized residuals."""
        return np.divide(v, sv)

    def _prepare_adjustment_results(self):
        """Prepare and store adjustment results."""
        n_measurements, n_unknowns = self._lsq_matrices.matrix_X.shape

        self._residual_variances = np.array(self._residual_variances)
        self._coord_corrections_variances = np.array(self._coord_corrections_variances)

        self._residual_sigmas = np.sqrt(self._residual_variances)
        self._coord_corrections_sigmas = np.sqrt(self._coord_corrections_variances)

        self._results = {
            "n_iter": self._iteration._counter,
            "n_measurements": n_measurements,
            "n_coord_corrections": self._coord_corrections.shape[0],
            "n_unknowns": n_unknowns,
            "degrees_of_freedom": self._lsq_matrices.degrees_of_freedom,
            "inner_constraints": self._lsq_matrices.inner_constraints,
            "residual_sigmas": self._residual_sigmas,
            "coord_correction_sigmas": self._coord_corrections_sigmas,
            "approximate_coordinates": self._approx_coordinates,
            "adjusted_coordinates": self._controls,
            "coordinate_corrections": self._coord_corrections,
            "obs_residuals": self._iteration.obs_residuals,
            "cov_X": self._iteration.covariance_X,
            "cov_Y": self._iteration.covariance_Y,
            "cov_r": self._iteration.covariance_r,
        }

    def solve(self):
        """Run the adjustment process."""
        self.iterate()
        print(self._iteration.counter)
        return self._check_condition()

    def iterate(self):
        """Perform a single iteration of the adjustment."""
        if self._iteration.counter > 1:
            self.update_matrices()
        self._iteration.run()
        self._update_coord_corrections()
        self.update_controls()
        self._append_residual_variances()

    def update_matrices(self):
        """Update X, Y, and weight matrices."""
        self._lsq_matrices.update_xy_matrices()
        self.update_weight_matrices()

    def update_weight_matrices(self):
        """Update weight matrices if tuning constants are present."""
        if self._lsq_matrices.tuning_constants:
            self.update_w_matrix()
        if self._lsq_matrices.free_tuning_constants:
            self.update_sw_matrix()

    def update_w_matrix(self):
        """Update observation weight matrix."""
        obs_v = self._iteration.obs_residuals.reshape(-1)
        obs_sv = self._iteration.covariance_r.diagonal()
        v = self._normalize_v(obs_v, obs_sv)

        if self._lsq_matrices.method == "cra":
            sigma_sq = self._iteration.residual_variance
            self._lsq_matrices.tuning_constants["sigma_sq"] = sigma_sq

        self._lsq_matrices.update_w_matrix(v)

    def update_sw_matrix(self):
        """Update control point weights matrix for free adjustment."""
        ctrl_v = self._iteration.increments.reshape(-1)
        ctrl_sv = self._iteration.covariance_X.diagonal()
        v = self._normalize_v(ctrl_v, ctrl_sv)

        if self._lsq_matrices.free_adjustment == "cra":
            coord_sigma_sq = self._coord_corrections_variances[-1]
            self._lsq_matrices.free_tuning_constants["sigma_sq"] = coord_sigma_sq

        self._lsq_matrices.update_sw_matrix(v)

    def update_controls(self):
        """Update control coordinates with increments."""
        controls = self._controls
        controls.loc[
            :, controls.coordinate_columns
        ] += self._iteration.increments_matrix
