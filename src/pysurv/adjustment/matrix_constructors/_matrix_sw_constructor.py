# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

from abc import abstractmethod

import numpy as np

from pysurv.data.dataset import Dataset

from ._matrix_x_indexer import MatrixXIndexer
from ._matrix_xyw_sw_constructor import MatrixXYWsWConstructor


class MatrixSWConstructor(MatrixXYWsWConstructor):
    """
    Abstract base class for constructing the controls weight matrix (sW) for adjustment computations.

    This class provides the interface and common functionality for building the diagonal controls weight
    matrix (sW).
    """

    def __init__(
        self,
        dataset: Dataset,
        matrix_x_indexer: MatrixXIndexer,
        default_sigmas_index: str | None,
    ) -> None:
        super().__init__(dataset, matrix_x_indexer, default_sigmas_index)

    @abstractmethod
    def build(self, matrix_x_n_col: int):
        """Build and return the diagonal controls weight matrix (sW)."""
        pass

    def _initialize_sw_matrix(self, matrix_x_n_col: int):
        """Initialize and return sW vector of length matrix_x_n_col."""
        return np.zeros(matrix_x_n_col)
