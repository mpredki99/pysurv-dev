# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

from typing import Any

import pandas as pd

from pysurv.basic.basic import from_rad, to_rad
from pysurv.validators._models import (
    ControlPointModel,
    MeasurementModel,
    validate_sigma,
)

from ._constants import DEFAULT_CONFIG_SIGMA
from .adjustment_config import AdjustmentConfig, AdjustmentConfigRow

__all__ = ["config_sigma"]


class ConfigSigma(AdjustmentConfig):
    """
    SigmaConfig manages and stores sigma (standard deviation) configurations for survey adjustment.

    This class allows you to define, access, and modify multiple sets of sigma values,
    which are used in adjustment computations. Each set is identified by a unique index,
    and one set is always designated as the default.
    """

    _default = DEFAULT_CONFIG_SIGMA

    def __getitem__(self, name: str):
        try:
            return super().__getitem__(name)
        except AttributeError:
            raise AttributeError(f"Config sigma has no index or column: {name}")

    def __str__(self) -> str:
        """Return a string representation of the sigma configuration"""
        text = "----- CONFIG SIGMA -----" + "\n"
        text += f"default index: {self.default_index}" + "\n\n"
        text += self._dataframe.to_string()
        return text

    def _get_kwarg(self, kwargs: dict, key: str, angle_unit: str | None = None):
        """Get a value from kwargs or default config."""
        value = kwargs.get(key)
        return (
            value if value is not None else self.default.get(key, angle_unit=angle_unit)
        )

    def restore_default(self) -> None:
        """Restore the default config sigma row to its initial state."""
        self.__setattr__("default", SigmaRow(angle_unit="rad", **self._default))

    def append(self, name: str | None = None, angle_unit: str | None = None, **kwargs):
        """Append a new sigma row to the config sigma."""
        data = {
            key: self._get_kwarg(kwargs, key, angle_unit=angle_unit)
            for key in self._default.keys()
        }
        name = self._validate_name(name)
        if name in self.index:
            raise IndexError(f"Given index name already exist: {name}")
        data["name"] = name

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
            raise IndexError(f"Sigma config does not have index: {index}")

        data = self.display(angle_unit=angle_unit)
        return data.loc[index]


class SigmaRow(AdjustmentConfigRow):
    """
    SigmaRow represents a single row of sigma (standard deviation) configuration values.
    """

    def __init__(self, angle_unit: str | None = "rad", **kwargs) -> None:
        data = {}
        name = kwargs.pop("name", "default")
        for key, value in kwargs.items():
            data[key] = self._validate_attr(key, value, angle_unit=angle_unit)

        self._data = pd.Series(data, name=name)

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
        return validate_sigma(value)

    def _validate_angle_sigma(self, value: float, angle_unit: str | None = "rad"):
        """Validate angle sigma value."""
        value = validate_sigma(value)
        if angle_unit == "rad":
            return value
        return to_rad(value, unit=angle_unit)

    def _validate_control_point_sigma(self, value: float):
        """Validate control point sigma value."""
        return validate_sigma(value, enable_minus_one=True)

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


config_sigma = ConfigSigma()
