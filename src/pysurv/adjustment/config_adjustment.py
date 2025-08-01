# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

from abc import ABC, abstractmethod
from typing import Any
from warnings import warn

import pandas as pd

from pysurv.warnings._warnings import DefaultIndexWarning


class ConfigAdjustment(ABC):
    def __init__(self) -> None:
        self._default_index = "default"
        self.restore_default()

    def __getitem__(self, name: str):
        """Get config row passing index in square bracets."""
        if name in self.index:
            return self.__getattribute__(name)
        elif name in self._dataframe.columns:
            return self._dataframe[name]
        else:
            raise AttributeError(f"Did not found index or column: {name}")

    def __delattr__(self, name: str) -> None:
        """Delete a sigma row by name, with protection for 'default' index."""
        if name == "default":
            raise ValueError("Can not delete 'default' index.")
        elif name == self._default_index:
            self._default_index = "default"
        super().__delattr__(name)

    def __delattr__(self, name: str) -> None:
        """Delete a config row by name, with protection for 'default' index."""
        if name == "default":
            raise ValueError("Can not delete 'default' index.")
        elif name == self._default_index:
            self._default_index = "default"
        super().__delattr__(name)

    @property
    def _dataframe(self) -> pd.DataFrame:
        """Return a DataFrame of representation of config object."""
        data = pd.DataFrame(
            {
                idx: row._data
                for idx, row in self.__dict__.items()
                if idx != "_default_index"
            }
        )
        return data.T

    def _validate_name(self, name: str | None) -> str:
        """Validate a name for a config row."""
        if name is None:
            return f"index_{len(self.index) - 1}"
        elif name.strip().isidentifier():
            return name.strip()
        raise ValueError("Attribute name is not valid identifier.")

    @property
    def index(self) -> pd.Index:
        """Returns the name index of all config rows."""
        return pd.Index(idx for idx in self.__dict__.keys() if idx != "_default_index")

    @property
    def columns(self) -> pd.Index:
        """Returns sigma columns names as pandas index."""
        return self._dataframe.columns

    @property
    def default_index(self) -> str:
        """Return the name of the default config row index."""
        return self._default_index

    @default_index.setter
    def default_index(self, name: str) -> None:
        """Set the name of the default sigma row index."""
        if name not in self.__dict__.keys():
            raise ValueError(f"Do not has index: {name}")
        self._default_index = name

    @default_index.deleter
    def default_index(self) -> None:
        """Reset default_index to 'default' on delete."""
        warn(
            "Can not delete default_index attribute. Setted 'default' instead.",
            DefaultIndexWarning,
        )
        self._default_index = "default"

    @abstractmethod
    def __str__(self) -> str:
        """Return a string representation of the config."""
        pass

    @abstractmethod
    def _get_kwarg(self, kwargs: dict, key: str) -> float | int:
        """Get a value from kwargs or default config."""
        pass

    @abstractmethod
    def restore_default(self) -> None:
        """Restore the default config row to its initial state."""
        pass

    @abstractmethod
    def append(self, name: str | None = None, **kwargs) -> None:
        """Append a new row to the config."""
        pass


class ConfigRow(ABC):
    """Base class for config rows."""

    def __getattr__(self, name: str):
        """Get sigma value from the sigma row."""
        if name == "_data":
            return super().__getattr__(name)
        return self._data.get(name)

    def __getitem__(self, name: str) -> float:
        """Get sigma value by name."""
        return self._data[name]

    def __setattr__(self, name: str, value: Any) -> None:
        """Set _data (row storing attribute) or set row sigma values with validation."""
        if name == "_data":
            super().__setattr__(name, value)
        else:
            self._data[name] = self._validate_attr(name, value)

    def __str__(self) -> str:
        """Return string representation of the sigma row."""
        return self._data.__str__()

    @abstractmethod
    def _validate_attr(self, name: str, value: Any) -> Any:
        """Validate and return sigma value."""
        pass
