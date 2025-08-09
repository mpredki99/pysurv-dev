# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

from pysurv import Dataset
from pysurv.adjustment import Solver
from pysurv.adjustment.adjustment_matrices import AdjustmentMatrices


def test_prepare_results(
    adjustment_test_matrices: AdjustmentMatrices, adjustment_test_dataset: Dataset
) -> None:
    """Test that beofre start calculations results are not generated."""
    controls = adjustment_test_dataset.controls
    solver = Solver(controls, adjustment_test_matrices)

    assert solver.n_iter == 0
    assert solver.results is None


def test_iterate(
    adjustment_test_matrices: AdjustmentMatrices, adjustment_test_dataset: Dataset
) -> None:
    """Test that iterate method works properly."""
    controls = adjustment_test_dataset.controls
    solver = Solver(controls, adjustment_test_matrices)

    solver.iterate()

    assert solver.n_iter == 1
    assert solver.results is not None


def test_solve_observation_ordinary(
    adjustment_test_matrices: AdjustmentMatrices, adjustment_test_dataset: Dataset
) -> None:
    """Test that solve method works properly with observation ordinary method."""
    controls = adjustment_test_dataset.controls
    matrices = adjustment_test_matrices
    matrices.methods.obs_adj = "ordinary"
    solver = Solver(controls, adjustment_test_matrices)

    assert solver.results is None

    solver.solve()

    assert solver.results.get("obs_adj_method") == "ordinary"
    assert solver.results.get("inner_constraints") is None
    assert solver.results is not None


def test_solve_observation_weighted(
    adjustment_test_matrices: AdjustmentMatrices, adjustment_test_dataset: Dataset
) -> None:
    """Test that solve method works properly with observation weighted method."""
    controls = adjustment_test_dataset.controls
    matrices = adjustment_test_matrices
    matrices.methods.obs_adj = "weighted"
    solver = Solver(controls, adjustment_test_matrices)

    assert solver.results is None

    solver.solve()

    assert solver.results.get("obs_adj_method") == "weighted"
    assert solver.results.get("inner_constraints") is None
    assert solver.results is not None


def test_solve_observation_robust(
    adjustment_test_matrices: AdjustmentMatrices, adjustment_test_dataset: Dataset
) -> None:
    """Test that solve method works properly with observation robust method."""
    controls = adjustment_test_dataset.controls
    matrices = adjustment_test_matrices
    matrices.methods.obs_adj = "huber"
    solver = Solver(controls, adjustment_test_matrices)

    assert solver.results is None

    solver.solve()

    assert solver.results.get("obs_adj_method") == "huber"
    assert solver.results.get("inner_constraints") is None
    assert solver.results is not None


def test_solve_free_adj_ordinary(
    adjustment_test_matrices: AdjustmentMatrices, adjustment_test_dataset: Dataset
) -> None:
    """Test that solve method works properly with free adjustment ordinary method."""
    controls = adjustment_test_dataset.controls
    matrices = adjustment_test_matrices
    matrices.methods.free_adjustment = "ordinary"
    solver = Solver(controls, adjustment_test_matrices)

    assert solver.results is None

    solver.solve()

    assert solver.results.get("free_adj_method") == "ordinary"
    assert solver.results.get("inner_constraints") == ["pseudoinverse"]
    assert solver.results is not None


def test_solve_free_adj_weighted(
    adjustment_test_matrices: AdjustmentMatrices, adjustment_test_dataset: Dataset
) -> None:
    """Test that solve method works properly with free adjustment weighted method."""
    controls = adjustment_test_dataset.controls
    matrices = adjustment_test_matrices
    matrices.methods.free_adjustment = "weighted"
    solver = Solver(controls, adjustment_test_matrices)

    assert solver.results is None

    solver.solve()

    assert solver.results.get("free_adj_method") == "weighted"
    assert solver.results.get("inner_constraints") is not None
    assert solver.results is not None


def test_solve_free_adj_robust(
    adjustment_test_matrices: AdjustmentMatrices, adjustment_test_dataset: Dataset
) -> None:
    """Test that solve method works properly with free adjustment robust method."""
    controls = adjustment_test_dataset.controls
    matrices = adjustment_test_matrices
    matrices.methods.free_adjustment = "huber"
    solver = Solver(controls, adjustment_test_matrices)

    assert solver.results is None

    solver.solve()

    assert solver.results.get("free_adj_method") == "huber"
    assert solver.results.get("inner_constraints") is not None
    assert solver.results is not None
