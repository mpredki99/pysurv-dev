# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

import numpy as np

from ._matrix_constructor import MatrixConstructor


class InnerConstraintsConstructor(MatrixConstructor):
    """
    Abstract base class for constructing inner contraints R matrix, list of that constraints
    and control point corrections matrix sX.

    This class initialize matrices constructor with specified number of columns in design matrix X.
    """

    def __init__(self, dataset, matrix_x_indexer, matrix_x_n_col: int) -> None:
        super().__init__(dataset, matrix_x_indexer)
        self._matrix_x_n_col = matrix_x_n_col
