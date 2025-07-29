# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

from pysurv.adjustment.adjustment import Adjustment
from pysurv.adjustment.lsq_matrices import LSQMatrices
from pysurv.adjustment.lsq_solver import LSQSolver
from pysurv.data.dataset import Dataset

from .config import config


class Project:
    """
    Root class for managing a pysurv project, including data, configuration, and adjustment operations.

    This class provides a unified interface for handling the entire workflow from data management to
    least squares adjustment and result reporting.
    """

    def __init__(self, dataset: Dataset) -> None:
        self._config = config
        self._dataset = dataset

        self._adjustment = None

    @property
    def adjustment(self):
        """Return the adjustment object."""
        return self._adjustment

    def adjust(
        self,
        method: str = "weighted",
        tuning_constants: dict | None = None,
        free_adjustment: str | None = None,
        free_tuning_constants: dict | None = None,
        default_sigmas_index: str | None = None,
        matrices_build_strategy: str | None = None,
        solve_strategy: str | None = None,
    ) -> None:
        """Perform least squares adjustment."""
        lsq_matrices = LSQMatrices(
            self._dataset,
            method=method,
            tuning_constants=tuning_constants,
            free_adjustment=free_adjustment,
            free_tuning_constants=free_tuning_constants,
            default_sigmas_index=default_sigmas_index,
            build_strategy=matrices_build_strategy,
        )
        lsq_solver = LSQSolver(
            self._dataset.controls, lsq_matrices, solve_strategy=solve_strategy
        )
        self._adjustment = Adjustment(lsq_solver)
        self._adjustment.adjust()
