# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

from pysurv.data.dataset import Dataset

from ._matrix_x_indexer import MatrixXIndexer
from ._matrix_xyw_sw_strategy import MatrixXYWsWStrategy
from ._speed_sw_constructor import SpeedSWConstructor
from ._speed_xyw_constructor import SpeedXYWConstructor


class SpeedStrategy(MatrixXYWsWStrategy):
    """
    SpeedStrategy implements a fast, vectorized strategy for constructing the design matrix (X),
    observation vector (Y), observation weights (W), and control weights (sW) for least squares
    adjustment in surveying networks.
    """

    def __init__(
        self,
        dataset: Dataset,
        matrix_x_indexer: MatrixXIndexer,
        default_sigmas_index: str | None,
    ) -> None:
        self._xyw_builder = SpeedXYWConstructor(
            dataset, matrix_x_indexer, default_sigmas_index
        )
        self._sw_builder = SpeedSWConstructor(
            dataset, matrix_x_indexer, default_sigmas_index
        )

    @property
    def xyw_constructor(self):
        """Returns speed constructor for X, Y, W matrices."""
        return self._xyw_builder

    @property
    def sw_constructor(self):
        """Returns speed constructor for sW matrix."""
        return self._sw_builder
