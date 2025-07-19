# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

from typing import Any
from warnings import warn

import pandas as pd

from pysurv.basic.basic import from_rad, to_rad
from pysurv.validators._models import (
    ControlPointModel,
    MeasurementModel,
    sigma_validator,
)
from pysurv.warnings._warnings import DefaultIndexWarning

from ._constants import DEFAULT_SIGMAS

__all__ = ["sigma_config"]


class SigmaConfig:
    """
    SigmaConfig manages and stores sigma (standard deviation) configurations for survey adjustment.

    This class allows you to define, access, and modify multiple sets of sigma values,
    which are used in adjustment computations. Each set is identified by a unique index,
    and one set is always designated as the default.
    """

    _default = DEFAULT_SIGMAS

    def __init__(self) -> None:
        self._default_index = "default"
        self.restore_default()

    def __delattr__(self, name: str) -> None:
        """Delete a sigma row by name, with protection for 'default' index."""
        if name == "default":
            raise ValueError("Can not delete 'default' index.")
        elif name == self._default_index:
            self._default_index = "default"
        super().__delattr__(name)

    def __getitem__(self, name: str):
        """Get sigma row by name."""
        return self.__getattribute__(name)

    def __repr__(self) -> pd.DataFrame:
        """Return a DataFrame representation of all sigma rows."""
        return self._dataframe

    def __str__(self) -> str:
        """Return a string representation of the sigma configuration"""
        text = "----- SIGMA CONFIG -----" + "\n"
        text += f"default index: {self.default_index}" + "\n\n"
        text += self._dataframe.to_string()
        return text

    @property
    def _dataframe(self) -> pd.DataFrame:
        """Return a DataFrame of all sigma rows."""
        data = pd.DataFrame(
            {
                idx: row._data
                for idx, row in self.__dict__.items()
                if idx != "_default_index"
            }
        )
        return data.T

    def _validate_name(self, name: str | None) -> str:
        """Validate a name for a sigma row."""
        if name is None:
            return f"index_{len(self.__dict__)}"
        elif name.strip().isidentifier():
            return name.strip()
        raise ValueError("Attribute name is not valid identifier.")

    @property
    def index(self) -> pd.Index:
        """Returns the name index of all sigma rows."""
        return pd.Index(self.__dict__.keys())

    @property
    def default_index(self) -> str:
        """Return the name of the default sigma row index."""
        return self._default_index

    @default_index.setter
    def default_index(self, name: str) -> None:
        """Set the name of the default sigma row index."""
        if name not in self.__dict__.keys():
            raise ValueError(f"Sigma config has no index: {name}")
        self._default_index = name

    @default_index.deleter
    def default_index(self):
        """Reset default_index to 'default' on delete."""
        warn(
            "Can not delete default_index property. Setted 'default' instead.",
            DefaultIndexWarning,
        )
        self._default_index = "default"

    def restore_default(self) -> None:
        """Restore the default sigma row to its initial state."""
        self.__setattr__("default", SigmaRow(angle_unit="rad", **self._default))

    def append(self, name: str | None = None, angle_unit: str | None = None, **kwargs):
        """Append a new sigma row to the sigma config."""
        data = {}
        for key in self._default.keys():
            data[key] = kwargs.get(key, self.default.get(key, angle_unit=angle_unit))
        name = self._validate_name(name)
        self.__setattr__(name, SigmaRow(angle_unit=angle_unit, **data))

    def display(self, angle_unit: str | None = None) -> pd.DataFrame:
        """Return a DataFrame of sigma values, converting angular units."""
        data = self._dataframe
        if angle_unit == "rad":
            return data
        anglular_columns = MeasurementModel.COLUMN_LABELS["angular_measurements_sigma"]
        data[anglular_columns] = from_rad(data[anglular_columns], unit=angle_unit)

        return data

    def get_row(self, index: str, angle_unit: str | None = None) -> pd.Series:
        """Return a sigma row as a pandas Series, converting angular units."""
        if index not in self.index:
            return IndexError(f"Sigma config does not have index: {index}")
        data = self.display(angle_unit=angle_unit)

        return data.loc[index]


class SigmaRow:
    """
    SigmaRow represents a single row of sigma (standard deviation) configuration values.
    """

    def __init__(self, angle_unit: str | None = "rad", **kwargs) -> None:
        data = {}
        for key, value in kwargs.items():
            data[key] = self._validate_attr(key, value, angle_unit=angle_unit)
        self._data = pd.Series(data)

    def __setattr__(self, name: str, value: Any) -> None:
        """Set _data (row storing attribute) or set row sigma values with validation."""
        if name == "_data":
            super().__setattr__(name, value)
        else:
            self._data[name] = self._validate_attr(name, value)

    def __getattr__(self, name: str):
        """Get sigma value from the sigma row."""
        if name == "_data":
            return super().__getattr__(name)
        return self._data.get(name)

    def __getitem__(self, name: str) -> float:
        """Get sigma value by name."""
        return self._data[name]

    def __str__(self) -> str:
        """Return string representation of the sigma row."""
        return self._data.__str__()

    def _validate_attr(
        self, name: str, value: float, angle_unit: str | None = "rad"
    ) -> float:
        """Validate and return sigma value."""
        if (
            name
            in MeasurementModel.COLUMN_LABELS["linear_measurements_sigma"]
            + MeasurementModel.COLUMN_LABELS["points_height_sigma"]
        ):
            return self._validate_distance_sigma(value)
        elif name in MeasurementModel.COLUMN_LABELS["angular_measurements_sigma"]:
            return self._validate_angle_sigma(value, angle_unit=angle_unit)
        elif name in ControlPointModel.COLUMN_LABELS["sigma"]:
            return self._validate_control_point_sigma(value)
        else:
            raise AttributeError(f"Sigma do not have attribute {name}")

    def _validate_distance_sigma(self, value: float) -> float:
        """Validate distance sigma value."""
        return sigma_validator(value)

    def _validate_angle_sigma(self, value: float, angle_unit: str | None = "rad"):
        """Validate angle sigma value."""
        value = sigma_validator(value)
        if angle_unit == "rad":
            return value
        return to_rad(value, unit=angle_unit)

    def _validate_control_point_sigma(self, value: float):
        """Validate control point sigma value."""
        return sigma_validator(value, enable_minus_one=True)

    def set(self, name: str, value: float, angle_unit: str | None = "rad"):
        """Set sigma value by name with angle unit specified."""
        self._data[name] = self._validate_attr(name, value, angle_unit=angle_unit)

    def get(self, name: str, angle_unit: str | None = "rad"):
        """Get sigma value by name with angle unit specified."""
        if name not in self._data.keys():
            raise AttributeError(f"Sigma does not have attribute {name}")
        if (
            angle_unit != "rad"
            and name in MeasurementModel.COLUMN_LABELS["angular_measurements_sigma"]
        ):
            return from_rad(self._data[name], unit=angle_unit)
        return self._data[name]


sigma_config = SigmaConfig()
