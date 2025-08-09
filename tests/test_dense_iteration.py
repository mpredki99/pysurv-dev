# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

from pysurv import Dataset
from pysurv.adjustment.adjustment_matrices import AdjustmentMatrices
from pysurv.adjustment.dense_iteration import DenseIteration


def test_calculate_normal_equations(
    adjustment_test_matrices: AdjustmentMatrices,
) -> None:
    """Test that normal equations calculations work properly."""
    iteration = DenseIteration(adjustment_test_matrices)
    iteration.run()

    assert iteration.matrix_G is not None
    assert iteration.cross_product is not None


def test_calculate_increment_matrix(
    adjustment_test_matrices: AdjustmentMatrices, adjustment_test_dataset: Dataset
) -> None:
    """Test that increment matrix calculations work properly."""
    iteration = DenseIteration(adjustment_test_matrices)
    iteration.run()

    assert iteration.increment_matrix is not None
    assert (
        iteration.increments_matrix.shape
        == adjustment_test_dataset.controls.coordinates.shape
    )


def test_calculate_increment_matrix(
    adjustment_test_matrices: AdjustmentMatrices,
) -> None:
    """Test that point weights calculations work properly."""
    iteration = DenseIteration(adjustment_test_matrices)
    iteration.run()

    assert iteration.coordinate_weights is not None


def test_calculate_increment_matrix(
    adjustment_test_matrices: AdjustmentMatrices,
) -> None:
    """Test that observation residuals calculations work properly."""
    iteration = DenseIteration(adjustment_test_matrices)
    iteration.run()

    assert iteration.obs_residuals is not None


def test_calculate_residual_variance(
    adjustment_test_matrices: AdjustmentMatrices,
) -> None:
    """Test that residual variance calculations work properly."""
    iteration = DenseIteration(adjustment_test_matrices)
    iteration.run()

    assert iteration.residual_variance is not None


def test_calculate_covariance_matrices(
    adjustment_test_matrices: AdjustmentMatrices,
) -> None:
    """Test that residual variance calculations work properly."""
    iteration = DenseIteration(adjustment_test_matrices)
    iteration.run()

    n_measurements, n_unknowns = adjustment_test_matrices.matrix_X.shape

    assert iteration.covariance_X is not None
    assert iteration.covariance_X.shape == (n_unknowns, n_unknowns)
    assert iteration.covariance_Y is not None
    assert iteration.covariance_Y.shape == (n_measurements, n_measurements)
    assert iteration.covariance_r is not None
    assert iteration.covariance_r.shape == (n_measurements, n_measurements)


def test_run(adjustment_test_matrices: AdjustmentMatrices) -> None:
    """Test that iteration runs properly."""
    iteration = DenseIteration(adjustment_test_matrices)

    assert iteration.counter == 0

    iteration.run()

    assert iteration.counter == 1
