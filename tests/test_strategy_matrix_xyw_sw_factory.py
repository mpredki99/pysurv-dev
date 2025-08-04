# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

from unittest.mock import Mock

import pandas as pd

from pysurv.adjustment.matrix_constructors.memory_strategy import MemoryStrategy
from pysurv.adjustment.matrix_constructors.speed_strategy import SpeedStrategy
from pysurv.adjustment.matrix_constructors.strategy_matrix_xyw_sw_factory import (
    get_strategy,
)


def test_matrices_speed_strategy_factory(mock_dataset_size) -> None:
    """Test that speed strategy is created if datasets size sum is less than 1 GB."""
    one_subset_size_GB = 1024**3 / 3
    size = pd.Series([one_subset_size_GB - 1])

    dataset = mock_dataset_size(size=size)
    strategy = get_strategy(
        dataset=dataset,
        matrix_x_indexer=Mock(return_value={}),
        default_sigmas_index=None,
    )

    assert isinstance(strategy, SpeedStrategy)


def test_matrices_memory_strategy_factory(mock_dataset_size) -> None:
    """Test that memory safe strategy is created if datasets size sum is greater or equal than 1 GB."""
    one_subset_size_GB = 1024**3 / 3
    size = pd.Series([one_subset_size_GB])

    dataset = mock_dataset_size(size=size)
    strategy = get_strategy(
        dataset=dataset,
        matrix_x_indexer=Mock(return_value={}),
        default_sigmas_index=None,
    )

    assert isinstance(strategy, MemoryStrategy)
