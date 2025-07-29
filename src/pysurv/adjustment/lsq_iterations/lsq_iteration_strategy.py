# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

from abc import ABC, abstractmethod


class LSQIterationStrategy(ABC):
    """Abstract base class for LSQ iteration strategy objects."""

    @abstractmethod
    def run(self):
        """Run the LSQ iteration."""
        pass
