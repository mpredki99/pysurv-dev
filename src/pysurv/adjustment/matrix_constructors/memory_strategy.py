# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

from pysurv.data.dataset import Dataset

from .indexer_matrix_x import IndexerMatrixX
from .memory_sw_constructor import MemorySWConstructor
from .memory_xyw_constructor import MemoryXYWConstructor
from .strategy_matrix_xyw_sw import MatrixXYWsWStrategy


class MemoryStrategy(MatrixXYWsWStrategy):
    """
    MemoryStrategy implements a memory-efficient row-wise strategy for constructing the design matrix (X),
    observation vector (Y), observation weights (W), and control weights (sW) for least squares
    adjustment in surveying networks.
    """

    def __init__(
        self,
        dataset: Dataset,
        matrix_x_indexer: IndexerMatrixX,
        default_sigmas_index: str | None,
    ) -> None:
        super().__init__()
        self._xyw_builder = MemoryXYWConstructor(
            dataset, matrix_x_indexer, default_sigmas_index
        )
        self._sw_builder = MemorySWConstructor(
            dataset, matrix_x_indexer, default_sigmas_index
        )

    @property
    def xyw_constructor(self):
        """Returns memory safe constructor for X, Y, W matrices."""
        return self._xyw_builder

    @property
    def sw_constructor(self):
        """Returns memory safe constructor for sW matrix."""
        return self._sw_builder
