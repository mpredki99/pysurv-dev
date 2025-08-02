# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

from pysurv.adjustment.adjustment import Adjustment
from pysurv.adjustment.matrices_dense import MatricesDense
from pysurv.adjustment.method_manager_robust import MethodManagerRobust
from pysurv.adjustment.solver import Solver
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
        config_sigma_index: str | None = None,
        matrices_build_strategy: str | None = None,
        config_solver_index: str | None = None,
    ) -> None:
        """Perform least squares adjustment."""
        method_manager = MethodManagerRobust(
            observations=method,
            obs_tuning_constants=tuning_constants,
            free_adjustment=free_adjustment,
            free_adj_tuning_constants=free_tuning_constants,
        )
        matrices = MatricesDense(
            self._dataset,
            method_manager,
            config_sigma_index=config_sigma_index,
            build_strategy=matrices_build_strategy,
        )
        solver = Solver(
            self._dataset.controls, matrices, config_solver_index=config_solver_index
        )
        self._adjustment = Adjustment(solver)
        return self._adjustment.solver.solve()
