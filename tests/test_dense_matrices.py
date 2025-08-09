# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

from pysurv import Dataset
from pysurv.adjustment import DenseMatrices
from pysurv.adjustment.adjustment_method_manager import AdjustmentMethodManager


def test_lazy_loading(
    MethodManagerTester: AdjustmentMethodManager, adjustment_test_dataset: Dataset
):
    """Test that matrices are not initialized while creating matrices object."""
    methods = MethodManagerTester()
    matrices = DenseMatrices(adjustment_test_dataset, methods)

    assert matrices._X is None
    assert matrices._Y is None
    assert matrices._W is None
    assert matrices._sW is None
    assert matrices._inner_constraints is None
    assert matrices._R is None
    assert matrices._sX is None
    assert matrices._k is None


def test_matrices_creation_with_ordinary_obs_method(
    MethodManagerTester: AdjustmentMethodManager,
    adjustment_test_dataset: Dataset,
    strategies: list,
):
    """Test that proper matrices are created with ordinary observations method."""
    methods = MethodManagerTester(obs_adj="ordinary", free_adjustment=None)
    for strategy in strategies:
        matrices = DenseMatrices(
            adjustment_test_dataset, methods, build_strategy=strategy
        )

        assert matrices.matrix_X is not None
        assert matrices.matrix_Y is not None
        assert matrices.matrix_W is None
        assert matrices.matrix_sW is not None
        assert matrices.inner_constraints is None
        assert matrices.matrix_R is None
        assert matrices.matrix_sX is not None
        assert matrices.degrees_of_freedom is not None


def test_matrices_creation_with_weighted_obs_method(
    MethodManagerTester: AdjustmentMethodManager,
    adjustment_test_dataset: Dataset,
    strategies: list,
):
    """Test that proper matrices are created with weighted observations method."""
    methods = MethodManagerTester(obs_adj="weighted", free_adjustment=None)
    for strategy in strategies:
        matrices = DenseMatrices(
            adjustment_test_dataset, methods, build_strategy=strategy
        )

        assert matrices.matrix_X is not None
        assert matrices.matrix_Y is not None
        assert matrices.matrix_W is not None
        assert matrices.matrix_sW is not None
        assert matrices.inner_constraints is None
        assert matrices.matrix_R is None
        assert matrices.matrix_sX is not None
        assert matrices.degrees_of_freedom is not None


def test_matrices_creation_with_robust_obs_method(
    MethodManagerTester: AdjustmentMethodManager,
    adjustment_test_dataset: Dataset,
    strategies: list,
):
    """Test that proper matrices are created with robust observations method."""
    methods = MethodManagerTester(obs_adj="huber", free_adjustment=None)
    for strategy in strategies:
        matrices = DenseMatrices(
            adjustment_test_dataset, methods, build_strategy=strategy
        )

        assert matrices.matrix_X is not None
        assert matrices.matrix_Y is not None
        assert matrices.matrix_W is not None
        assert matrices.matrix_sW is not None
        assert matrices.inner_constraints is None
        assert matrices.matrix_R is None
        assert matrices.matrix_sX is not None
        assert matrices.degrees_of_freedom is not None


def test_matrices_creation_with_ordinary_free_adj_method(
    MethodManagerTester: AdjustmentMethodManager,
    adjustment_test_dataset: Dataset,
    strategies: list,
):
    """Test that proper matrices are created with ordinary free adjustment method."""
    methods = MethodManagerTester(obs_adj="ordinary", free_adjustment="ordinary")
    for strategy in strategies:
        matrices = DenseMatrices(
            adjustment_test_dataset, methods, build_strategy=strategy
        )

        assert matrices.matrix_X is not None
        assert matrices.matrix_Y is not None
        assert matrices.matrix_W is None
        assert matrices.matrix_sW is None
        assert matrices.inner_constraints is None
        assert matrices.matrix_R is None
        assert matrices.matrix_sX is None
        assert matrices.degrees_of_freedom is not None


def test_matrices_creation_with_weighted_free_adj_method(
    MethodManagerTester: AdjustmentMethodManager,
    adjustment_test_dataset: Dataset,
    strategies: list,
):
    """Test that proper matrices are created with weighted free adjustment method."""
    methods = MethodManagerTester(obs_adj="ordinary", free_adjustment="weighted")
    for strategy in strategies:
        matrices = DenseMatrices(
            adjustment_test_dataset, methods, build_strategy=strategy
        )

        assert matrices.matrix_X is not None
        assert matrices.matrix_Y is not None
        assert matrices.matrix_W is None
        assert matrices.matrix_sW is not None
        assert matrices.inner_constraints is not None
        assert matrices.matrix_R is not None
        assert matrices.matrix_sX is None
        assert matrices.degrees_of_freedom is not None


def test_matrices_creation_with_robust_free_adj_method(
    MethodManagerTester: AdjustmentMethodManager,
    adjustment_test_dataset: Dataset,
    strategies: list,
):
    """Test that proper matrices are created with weighted free adjustment method."""
    methods = MethodManagerTester(obs_adj="ordinary", free_adjustment="weighted")
    for strategy in strategies:
        matrices = DenseMatrices(
            adjustment_test_dataset, methods, build_strategy=strategy
        )

        assert matrices.matrix_X is not None
        assert matrices.matrix_Y is not None
        assert matrices.matrix_W is None
        assert matrices.matrix_sW is not None
        assert matrices.inner_constraints is not None
        assert matrices.matrix_R is not None
        assert matrices.matrix_sX is None
        assert matrices.degrees_of_freedom is not None


def test_matrices_inner_constraints_changed(
    MethodManagerTester: AdjustmentMethodManager, adjustment_test_dataset: Dataset
):
    """Test that inner constraint matrix is not return after changing into proper method."""
    methods = MethodManagerTester(obs_adj="ordinary", free_adjustment="weighted")
    matrices = DenseMatrices(adjustment_test_dataset, methods)

    assert matrices.matrix_sW is not None
    assert matrices.inner_constraints is not None
    assert matrices.matrix_R is not None

    methods.free_adjustment = "ordinary"

    assert matrices.matrix_sW is None
    assert matrices.inner_constraints is None
    assert matrices.matrix_R is None


def test_matrix_sx_changed(
    MethodManagerTester: AdjustmentMethodManager, adjustment_test_dataset: Dataset
):
    """Test that sX matrix is not return after changing into proper method."""
    methods = MethodManagerTester(obs_adj="ordinary", free_adjustment=None)
    matrices = DenseMatrices(adjustment_test_dataset, methods)

    assert matrices.matrix_sX is not None

    methods.free_adjustment = "weighted"

    assert matrices.matrix_sX is None


def test_matrix_w_changed(
    MethodManagerTester: AdjustmentMethodManager, adjustment_test_dataset: Dataset
):
    """Test that W matrix is not return after changing into proper method."""
    methods = MethodManagerTester(obs_adj="weighted", free_adjustment=None)
    matrices = DenseMatrices(adjustment_test_dataset, methods)

    assert matrices.matrix_W is not None

    methods.obs_adj = "ordinary"

    assert matrices.matrix_W is None
