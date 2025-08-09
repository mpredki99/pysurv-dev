# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

from pysurv.data.dataset import Dataset

from .dense_matrices import DenseMatrices
from .method_manager import MethodManager
from .report import Report
from .solver import Solver


class Adjustment:
    """
    Highest level adjustment module class for running least squares adjustment
    and show results.
    """

    def __init__(
        self,
        dataset: Dataset,
        obs_adj: str = "weighted",
        obs_tuning_constants: dict | None = None,
        free_adjustment: str | None = None,
        free_adj_tuning_constants: dict | None = None,
        config_sigma_index: str | None = None,
        matrices_build_strategy: str | None = None,
        config_solver_index: str | None = None,
        create_list_of_variances: bool = False,
    ) -> None:
        self._dataset = dataset
        method_manager = MethodManager(
            obs_adj=obs_adj,
            obs_tuning_constants=obs_tuning_constants,
            free_adjustment=free_adjustment,
            free_adj_tuning_constants=free_adj_tuning_constants,
        )
        matrices = DenseMatrices(
            self._dataset,
            method_manager,
            config_sigma_index=config_sigma_index,
            build_strategy=matrices_build_strategy,
        )
        self._solver = Solver(
            self._dataset.controls,
            matrices,
            config_solver_index=config_solver_index,
            create_list_of_variances=create_list_of_variances,
        )
        self._report = None

    @property
    def solver(self):
        return self._solver

    @property
    def matrices(self):
        return self._solver.matrices

    @property
    def methods(self):
        return self.matrices.methods

    @property
    def dataset(self):
        return self._dataset

    @property
    def report(self):
        """Return the adjustment report."""
        if self._solver.results is not None:
            self._report = Report(self._solver.results)
        return self._report
