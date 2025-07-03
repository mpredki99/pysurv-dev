import numpy as np

from pysurv.config import config
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
    
    
def azimuth(x_first, y_first, x_second, y_second):
    dx = x_second - x_first
    dy = y_second - y_first

    if dx == 0 and dy == 0:
        raise ValueError('Start and end points overlap. Azimuth for point is indefinite.')

    return np.arctan2(dy, dx) % (2 * np.pi)
