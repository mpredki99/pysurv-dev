# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

__all__ = ["config_solver"]


class ConfigSolver:
    """Configuration class for solver iteration threshold and maximum iterations."""

    def __init__(self) -> None:
        self._threshold = 0.001
        self._max_iter = 100

    @property
    def threshold(self) -> float:
        """Get the iteration threshold."""
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        """Set the iteration threshold."""
        try:
            self._threshold = float(value)
        except ValueError:
            raise ValueError(
                f"Iteration threshold shold be float or possible to convert: {value}"
            )

    @property
    def max_iter(self) -> int:
        """Get the maximum number of iterations."""
        return self._max_iter

    @max_iter.setter
    def max_iter(self, value: int) -> None:
        """Set the maximum number of iterations."""
        try:
            self._max_iter = int(value)
        except ValueError:
            raise ValueError(
                f"Max iteration number shold be int or possible to convert: {value}"
            )


config_solver = ConfigSolver()
