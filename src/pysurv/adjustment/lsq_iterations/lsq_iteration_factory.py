# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

from ..lsq_matrices import LSQMatrices
from .dense_iteration import DenseIteration


def get_lsq_iteration(name: str | None, lsq_matrices: LSQMatrices):
    """Initialize and return LSQ iteration strategy object"""
    lsq_iteration = DenseIteration(lsq_matrices)

    return lsq_iteration
