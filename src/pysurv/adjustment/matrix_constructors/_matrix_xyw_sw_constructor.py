# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

from pysurv.data.dataset import Dataset

from ..sigma_config import sigma_config
from ._matrix_constructor import MatrixConstructor
from ._matrix_x_indexer import MatrixXIndexer


class MatrixXYWsWConstructor(MatrixConstructor):
    """
    Abstract base class for constructing X, Y, W, and sW matrices for adjustment computations.

    This class initialize matrices constructor with specified default sigmas row from sigma_config
    object.
    """

    def __init__(
        self,
        dataset: Dataset,
        matrix_x_indexer: MatrixXIndexer,
        default_sigmas_index: str,
    ) -> None:
        super().__init__(dataset, matrix_x_indexer)

        default_sigmas_index = default_sigmas_index or sigma_config.default_index
        self._default_sigmas = sigma_config[default_sigmas_index]
