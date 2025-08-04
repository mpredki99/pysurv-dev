# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

from abc import ABC, abstractmethod

from .matrices import Matrices


class Iteration(ABC):
    """Abstract base class for LSQ iteration strategy objects."""

    def __init__(self, matrices: Matrices) -> None:
        self._lsq_matrices = matrices

    @abstractmethod
    def run(self):
        """Run the LSQ iteration."""
        pass
