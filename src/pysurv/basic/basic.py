# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

import numpy as np

from pysurv.validators._validators import validate_angle_unit


def to_rad(angle: float, unit: str | None = None) -> float:
    """Convert angle from degrees or gradinas to radians."""
    unit = validate_angle_unit(unit)

    if unit in ["grad", "gon"]:
        return angle * np.pi / 200
    elif unit == "deg":
        return angle * np.pi / 180
    else:
        return angle


def from_rad(angle: float, unit: str | None = None) -> float:
    """Convert angle from radians to degrees or gradinas."""
    unit = validate_angle_unit(unit)

    if unit in ["grad", "gon"]:
        return angle * 200 / np.pi
    elif unit == "deg":
        return angle * 180 / np.pi
    else:
        return angle


def azimuth(x_first: float, y_first: float, x_second: float, y_second: float) -> float:
    """Calculate the azimuth in radians from coordinates."""
    dx = x_second - x_first
    dy = y_second - y_first

    return np.mod(np.arctan2(dy, dx), 2 * np.pi)
