# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

import pandas as pd

from pysurv.exceptions import InvalidAngleUnitError


def sigma_validator(
    v: float, 
    enable_minus_one: bool = False, 
    error_message: str = "Sigma values must be >= 0."
) -> float:
    """Validate and return sigma value or raise error with appropriate message."""
    is_empty = pd.isna(v)
    is_negative = not v >= 0

    error_condition = not is_empty and is_negative

    if enable_minus_one:
        error_condition = error_condition and not v == -1
        error_message = "Control point sigma values must be >= 0 or -1."

    if error_condition:
        raise ValueError(f"{error_message} Got {v}.")
    return v


def validate_angle_unit(v: str | None) -> str:
    """Validate and return angle unit. If None return value from config object."""
    if v is None:
        from pysurv.config import config
        v = config.angle_unit
    if v not in ["rad", "grad", "gon", "deg"]:
        raise InvalidAngleUnitError(
            "Angle unit must be either 'rad', 'grad', 'gon', 'deg'."
        )
    return v
