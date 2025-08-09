# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.


from functools import cached_property
from warnings import warn

import numpy as np

from pysurv.adjustment.matrices import Matrices
from pysurv.data.controls import Controls
from pysurv.warnings._warnings import InvalidVarianceWarning

from .adjustment_solver import AdjustmentSolver
from .dense_iteration import DenseIteration


class Solver(AdjustmentSolver):
    """Class for solving surveying adjustment task."""
    @property
    def n_iter(self):
        return self._iteration.counter

    @property
    def matrix_G(self):
        return self._iteration.matrix_G

    @property
    def inv_matrix_G(self):
        """Return inverse of G matrix."""
        return self._iteration.inv_matrix_G

    @property
    def cross_product(self):
        """Rerurn cross product"""
        return self._iteration.cross_product

    @property
    def increments(self):
        """Return increments."""
        return self._iteration.increments

    @property
    def coord_increments(self):
        """Return fitered for just coordinate increments."""
        return self._iteration.coord_increments

    @property
    def increment_matrix(self):
        """Return increment matrix."""
        return self._iteration.increment_matrix

    @property
    def obs_residuals(self):
        """Return observation residuals."""
        return self._iteration.obs_residuals

    @property
    def residual_variance(self):
        return self._iteration.residual_variance

    @property
    def covariance_X(self):
        """Return the covariance matrix of X."""
        return self._iteration.covariance_X

    @property
    def covariance_Y(self):
        """Return the covariance matrix of Y."""
        return self._iteration.covariance_Y

    @property
    def covariance_r(self):
        """Return the covariance matrix of residuals."""
        return self._iteration.covariance_r

    @property
    def coordinate_weights(self):
        """Return the point weights."""
        return self._iteration.coordinate_weights

    @property
    def n_movable_tie_points(self):
        return self._n_movable_tie_points

    @cached_property
    def coord_cor_variance(self):
        if not self._iteration:
            return None

        if self._coord_corrections_variances:
            return self._coord_corrections_variances[-1]
        else:
            return self._get_coord_corrections_variance()

    @cached_property
    def coord_corrections(self):
        return self._controls.coordinates - self._approx_coordinates

    @cached_property
    def svd_converge(self):
        return self._iteration.run()

    def solve(self):
        """Run the adjustment process."""
        if not self.iterate():
            return self.results
        return self._check_condition()

    def iterate(self):
        """Perform a single iteration of the adjustment."""
        self._prepare_iteration()

        if not self.svd_converge:
            return False
        self._process_successful_iteration()
        return True

    def update_matrices(self):
        """Update X, Y, and weight matrices."""
        self._matrices.update_xy_matrices()
        self._update_weight_matrices()

    def _update_weight_matrices(self):
        """Update weight matrices if tuning constants are present."""
        if self.methods.obs_tuning_constants:
            self._update_w_matrix()
        if self.methods.free_adj_tuning_constants:
            self._update_sw_matrix()

    def _update_w_matrix(self):
        """Update observation weight matrix."""
        obs_residuals = self._iteration.obs_residuals.reshape(-1)
        obs_residuals_var = self._iteration.covariance_r.diagonal()

        v = self._normalize_residuals(obs_residuals, obs_residuals_var)
        self._matrices.update_w_matrix(v)

    def _update_sw_matrix(self):
        """Update control point weights matrix for free adjustment."""
        increments = self._iteration.increments.reshape(-1)
        increments_var = self._iteration.covariance_X.diagonal()

        v = self._normalize_residuals(increments, increments_var)
        self._matrices.update_sw_matrix(v)

    def _normalize_residuals(self, v: np.ndarray, var_v: np.ndarray) -> np.ndarray:
        """Return normalized residuals."""
        sv = self._calculate_sigma(var_v)
        return np.divide(v, sv, out=np.full_like(v, -np.inf), where=sv > 0)

    def _calculate_sigma(self, var_v: np.ndarray) -> np.ndarray:
        """Calculate sigma based on variance."""
        invalid = var_v[var_v < 0]
        if invalid.size > 0:
            warn(
                f"{invalid.size} negative variances occured in {self.n_iter}. iteration: {invalid}.",
                InvalidVarianceWarning,
            )
            var_v = np.clip(var_v, a_min=0, a_max=None)
        return np.sqrt(var_v)

    def _prepare_iteration(self):
        """Prepare matrices for iteration."""
        if self._iteration:
            self.update_matrices()

    def _process_successful_iteration(self):
        """Process the results of a successful iteration."""
        self._reset_cache()
        self._update_controls()
        if self._create_list_of_variances:
            self._append_residual_variances()
        self._refresh_tuning_constants()

    def _refresh_tuning_constants(self):
        methods_to_update = ["t", "cra"]
        if self.methods.obs_adj in methods_to_update:
            self.methods._refresh_obs_tuning_constants()
        if self.methods.free_adjustment in methods_to_update:
            self.methods._refresh_free_tuning_constants()

    def _update_controls(self):
        """Update control coordinates with increments."""
        controls = self._controls
        controls.loc[:, controls.coordinate_columns] += self._iteration.increment_matrix

    def _append_residual_variances(self):
        """Append current residual and coordinate correction variances."""
        self._residual_variances.append(self.residual_variance)
        self._coord_corrections_variances.append(self.coord_cor_variance)

    def _check_condition(self):
        """Check if iteration should stop or continue."""
        if self._is_increments_within_threshold() or self._is_max_iter_exceeded():
            return self.results
        return self.solve()

    def _is_max_iter_exceeded(self):
        """Check if current iteration number is less than max in config."""
        return self._iteration._counter >= self._config_solver.max_iter

    def _is_increments_within_threshold(self):
        """Check if all increments are less than threshold in config."""
        return all(self._iteration.coord_increments <= self._config_solver.threshold)

    def _get_coord_corrections_variance(self):
        """Get value of variance of coordinate corrections."""
        if self._n_movable_tie_points > 0:
            return self._calculate_coord_coorections_variance()
        return 1

    def _calculate_coord_coorections_variance(self):
        """Calculate variance of coordinate corrections."""
        point_weights = self._iteration.coordinate_weights
        coord_corrections = self.coord_corrections.values.reshape(-1)

        if point_weights is not None:
            squared_corrections = (coord_corrections**2 * point_weights).sum()
        else:
            squared_corrections = (coord_corrections**2).sum()

        return np.divide(squared_corrections, self._n_movable_tie_points)

    def _prepare_svd_converge(self):
        if self.svd_converge:
            return "Calculations succeed."
        return "Calculations aborted due to SVD did not converge."

    def _prepare_inner_constraints(self):
        if self.matrices.methods.free_adjustment == "ordinary":
            return ["pseudoinverse"]
        return self.matrices.inner_constraints

    def _prepare_residual_sigma(self):
        if self._create_list_of_variances:
            residual_variances = np.array(self._residual_variances)
            return np.sqrt(residual_variances)
        return np.sqrt(self.residual_variance)

    def _prepare_coord_correction_sigma(self):
        if self._create_list_of_variances:
            coord_corrections_variances = np.array(self._coord_corrections_variances)
            return np.sqrt(coord_corrections_variances)
        return np.sqrt(self._get_coord_corrections_variance())

    def _prepare_adjustment_results(self):
        """Prepare and store adjustment results."""
        n_measurements, n_unknowns = self.matrices.matrix_X.shape

        self._results = {
            "n_iter": self.n_iter,
            "obs_adj_method": self.methods.obs_adj,
            "obs_tuning_constants": self.methods.obs_tuning_constants,
            "free_adj_method": self.methods.free_adjustment,
            "free_adj_tuning_constants": self.methods.free_adj_tuning_constants,
            "n_measurements": n_measurements,
            "n_coord_corrections": self.coord_corrections.size,
            "n_unknowns": n_unknowns,
            "n_movable_tie_points": self.n_movable_tie_points,
            "degrees_of_freedom": self.matrices.degrees_of_freedom,
            "inner_constraints": self._prepare_inner_constraints(),
            "obs_residuals": self.obs_residuals,
            "residual_sigma": self._prepare_residual_sigma(),
            "coord_correction_sigma": self._prepare_coord_correction_sigma(),
            "approximate_coordinates": self._approx_coordinates,
            "adjusted_coordinates": self._controls.coordinates,
            "coordinate_corrections": self.coord_corrections,
            "cov_X": self.covariance_X,
            "cov_Y": self.covariance_Y,
            "cov_r": self.covariance_r,
            "SVD_converge": self._prepare_svd_converge(),
        }

    def _reset_cache(self):
        """Reset cached properties values."""
        for name in dir(self.__class__):
            attr = getattr(self.__class__, name)
            if isinstance(attr, cached_property) and name in self.__dict__:
                del self.__dict__[name]
