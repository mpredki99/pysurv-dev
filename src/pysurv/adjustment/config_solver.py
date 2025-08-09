# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

import pandas as pd

from ._constants import DEFAULT_CONFIG_SOLVER
from .adjustment_config import AdjustmentConfig, AdjustmentConfigRow

__all__ = ["config_solver"]


class ConfigSolver(AdjustmentConfig):

    _default = DEFAULT_CONFIG_SOLVER

    def __getitem__(self, name: str):
        try:
            return super().__getitem__(name)
        except AttributeError:
            raise AttributeError(f"Config solver has no index or column: {name}")

    def __str__(self) -> str:
        """Return a string representation of the sigma configuration"""
        text = "----- CONFIG SOLVER -----" + "\n"
        text += f"default index: {self.default_index}" + "\n\n"
        text += self._dataframe.to_string()
        return text

    def _get_kwarg(self, kwargs: dict, key: str):
        """Get a value from kwargs or default config."""
        value = kwargs.get(key)
        return value if value is not None else self.default[key]

    def restore_default(self):
        """Restore the default config solver row to its initial state."""
        self.__setattr__("default", ConfigSolverRow(**self._default))

    def append(self, name: str | None = None, **kwargs):
        """Append a new sigma row to the config solver."""
        data = {key: self._get_kwarg(kwargs, key) for key in self._default.keys()}
        name = self._validate_name(name)
        if name in self.index:
            raise IndexError(f"Given index name already exist: {name}")
        data["name"] = name

        name = self._validate_name(name)
        if name in self.index:
            raise IndexError(f"Given index name already exist: {name}")

        data["name"] = name

        self.__setattr__(name, ConfigSolverRow(**data))


class ConfigSolverRow(AdjustmentConfigRow):
    """Configuration class for solver iteration threshold and maximum iterations."""

    def __init__(self, **kwargs) -> None:
        data = {}
        name = kwargs.pop("name", "default")
        for key, value in kwargs.items():
            data[key] = self._validate_attr(key, value)

        self._data = pd.Series(data, name=name)

    def _validate_attr(self, name: str, value: float | int) -> float:
        """Validate and return sigma value."""
        if name == "threshold":
            return self._validate_threshold(value)
        elif name == "max_iter":
            return self._validate_max_iter(value)
        else:
            raise AttributeError(f"Config Solver row do not have attribute {name}")

    def _validate_threshold(self, value: float) -> float:
        """Validate the iteration threshold."""
        try:
            value = float(value)
        except ValueError:
            raise ValueError(
                f"Iteration threshold shold be float or possible to convert: {value}"
            )
        return self.validate_positive(value, "Iteration threshold")

    def _validate_max_iter(self, value: int) -> int:
        """Validate the maximum number of iterations."""
        try:
            value = int(value)
        except ValueError:
            raise ValueError(
                f"Max iteration number shold be int or possible to convert: {value}"
            )
        return self.validate_positive(value, "Max iteration number")

    def validate_positive(self, value: float | int, message_prefix: str) -> float | int:
        """Validate that the value is positive."""
        if value <= 0:
            raise ValueError(f"{message_prefix} value must be positive: {value}")
        return value


config_solver = ConfigSolver()
