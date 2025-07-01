import numpy as np

from pysurv import config
from pysurv.models import validate_angle_unit


def to_rad(angle: float, unit: str | None = None) -> float:
    unit = config.angle_unit if unit is None else validate_angle_unit(unit)

    if unit in ["grad", "gon"]:
        return angle * np.pi / 200
    elif unit == "deg":
        return angle * np.pi / 180
    else:
        return angle


def from_rad(angle: float, unit: str | None = None) -> float:
    unit = config.angle_unit if unit is None else validate_angle_unit(unit)

    if unit in ["grad", "gon"]:
        return angle * 200 / np.pi
    elif unit == "deg":
        return angle * 180 / np.pi
    else:
        return angle
