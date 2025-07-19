# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

from .adjustment.sigma_config import SigmaConfig, sigma_config
from .validators._validators import validate_angle_unit

__all__ = ["config"]


class Config:
    """
    Global configuration for PySurv package.

    This module provides a singleton `config` object for managing global settings
    such as the default angle unit and default sigma values configuration used throughout the PySurv package.
    """

    def __init__(self) -> None:
        self._angle_unit: str = "grad"
        self._sigma_config: SigmaConfig = sigma_config

    @property
    def angle_unit(self) -> str:
        """Get the current default angle unit used in PySurv."""
        return self._angle_unit

    @angle_unit.setter
    def angle_unit(self, new_angle_unit: str | None) -> None:
        """Validate the value and set new default angle unit for used in PySurv."""
        new_angle_unit = validate_angle_unit(new_angle_unit)
        self._angle_unit = new_angle_unit

    @property
    def sigma_config(self) -> SigmaConfig:
        """Get the current default sigma configuration used in PySurv."""
        return self._sigma_config

    def __str__(self) -> str:
        """Return a string representation of the current global configuration."""
        text = "----- CONFIG -----" + "\n"
        text += f"Default angle unit: {self._angle_unit}" + "\n\n"
        for attr in self.__dict__.values():
            if attr == self._angle_unit:
                continue
            text += attr.__str__() + "\n"
        return text


config = Config()
