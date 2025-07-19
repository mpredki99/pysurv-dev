# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

from pysurv.data.dataset import Dataset
from pysurv.exceptions._exceptions import InvalidMethodError

from .lsq_matrices import LSQMatrices
from .robust import __all__ as robust_methods


class Adjustment:
    def __init__(
        self,
        dataset: Dataset,
        method: str = "weighted",
        free_adjustment: str | None = None,
        default_sigmas_index: str | None = None,
        computations_priority: str | None = None,
    ) -> None:
        self._dataset = dataset
        self._method = self._validate_method(method)

        if free_adjustment is None:
            self._free_adjustment = free_adjustment
        else:
            self._free_adjustment = self._validate_method(free_adjustment)

        self._lsq_matrices = LSQMatrices(
            self._dataset,
            calculate_weights=self._method != "ordinary",
            default_sigmas_index=default_sigmas_index,
            computations_priority=computations_priority,
        )

    def _validate_method(self, method: str):
        valid_methods = ["ordinary", "weighted"]
        valid_methods.extend(robust_methods)
        if method not in valid_methods:
            raise InvalidMethodError(
                f"Invalid weighting method. Valid methods: {valid_methods}"
            )
        return method

    @property
    def method(self) -> str:
        return self._method

    @property
    def free_adjustment(self) -> str | None:
        return self._free_adjustment
