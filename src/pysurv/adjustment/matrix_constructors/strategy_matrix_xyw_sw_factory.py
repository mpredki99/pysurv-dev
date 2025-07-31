# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

from pysurv.data.dataset import Dataset

from .._constants import MEMORY_THRESHOLD_GB
from .indexer_matrix_x import IndexerMatrixX
from .memory_strategy import MemoryStrategy
from .speed_strategy import SpeedStrategy
from .strategy_matrix_xyw_sw import MatrixXYWsWStrategy

"""
Module for selecting and instantiating matrix construction strategies for least squares adjustment.

This module provides a factory interface to choose between `fast` and `memory saving` matrix building 
strategies for the design matrix (X), observation vector (Y), and weight matrices (W, sW) used in 
surveying network adjustment computations. The strategy can be selected based on user input or
automatically determined by the dataset size.
"""


strategies = {"speed": SpeedStrategy, "memory_safe": MemoryStrategy}


def get_strategy(
    dataset: Dataset,
    matrix_x_indexer: IndexerMatrixX,
    default_sigmas_index: str,
    name: str | None = None,
) -> MatrixXYWsWStrategy:
    """Return an MatrixXYWsWStrategy instance by name."""
    global strategies
    name = get_strategy_name(name, dataset)

    strategy_constructor = strategies[name]

    return strategy_constructor(dataset, matrix_x_indexer, default_sigmas_index)


def get_strategy_name(name: str | None, dataset: Dataset):
    """Return the strategy name based on user input or dataset size."""
    if name is not None:
        return _validate_strategy_name(name)

    measurements_weight = dataset.measurements.memory_usage(deep=True).sum()
    controls_weight = dataset.controls.memory_usage(deep=True).sum()
    stations_weight = dataset.stations.memory_usage(deep=True).sum()

    total_weight = measurements_weight + controls_weight + stations_weight
    return "speed" if total_weight / (1024**3) < MEMORY_THRESHOLD_GB else "memory_safe"


def _validate_strategy_name(name: str):
    """Validate and return the strategy name."""
    global strategies

    strategies_list = list(strategies.keys())
    if name not in strategies_list:
        raise ValueError(f"Invalid strategy. Please choose one: {strategies_list}")

    return name
