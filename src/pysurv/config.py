# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

from .adjustment.config_sigma import ConfigSigma, config_sigma
from .adjustment.config_solver import ConfigSolver, config_solver
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
        self._config_sigma: ConfigSigma = config_sigma
        self._config_solver: ConfigSolver = config_solver

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
    def config_sigma(self) -> ConfigSigma:
        """Get the config sigma object ."""
        return self._config_sigma

    @property
    def config_solver(self) -> ConfigSolver:
        """Get the config solver object ."""
        return self._config_solver

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
