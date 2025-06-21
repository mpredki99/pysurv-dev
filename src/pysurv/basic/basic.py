import numpy as np


def to_rad(angle: float, unit: str = "grad") -> float:
    if unit in ["grad", "gon"]:
        return angle * np.pi / 200
    elif unit == "deg":
        return angle * np.pi / 180
    else:
        raise ValueError('Invalid unit. Use "grad", "gon", or "deg".')


def from_rad(angle: float, unit: str = "grad") -> float:
    if unit in ["grad", "gon"]:
        return angle * 200 / np.pi
    elif unit == "deg":
        return angle * 180 / np.pi
    else:
        raise ValueError('Invalid unit. Use "grad", "gon", or "deg".')
