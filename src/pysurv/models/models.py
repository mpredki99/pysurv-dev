from typing import ClassVar

import numpy as np
from pydantic import BaseModel, Field, field_validator


def validate_angle_unit(v):
    if v not in ["rad", "grad", "gon", "deg"]:
        raise ValueError("Angle unit must be either 'rad', 'grad', 'gon', 'deg'.")
    return v


def _validator(v, enable_minus_one=False, error_message="Sigma values must be >= 0."):
    is_empty = v is None or np.isnan(v)
    is_negative = not v >= 0

    error_condition = not is_empty and is_negative

    if enable_minus_one:
        error_condition = error_condition and not v == -1
        error_message = "Control point sigma values must be >= 0 or -1."

    if error_condition:
        raise ValueError(f"{error_message} Got {v}.")
    return v


class MeasurementModel(BaseModel):
    stn_pk: int
    trg_id: str
    trg_h: float | None = None
    trg_sh: float | None = Field(
        default=None, description="Standard deviation in trg_h."
    )
    sd: float | None = Field(default=None, description="Slope distance.")
    hd: float | None = Field(default=None, description="Horizontal distance.")
    vd: float | None = None
    dx: float | None = None
    dy: float | None = None
    dz: float | None = None
    ssd: float | None = Field(default=None, description="Standard deviation in sd.")
    shd: float | None = Field(default=None, description="Standard deviation in hd.")
    svd: float | None = Field(default=None, description="Standard deviation in vd.")
    sdx: float | None = Field(default=None, description="Standard deviation in dx.")
    sdy: float | None = Field(default=None, description="Standard deviation in dy.")
    sdz: float | None = Field(default=None, description="Standard deviation in dz.")
    a: float | None = None
    hz: float | None = None
    vz: float | None = None
    vh: float | None = None
    sa: float | None = Field(default=None, description="Standard deviation in a.")
    shz: float | None = Field(default=None, description="Standard deviation in hz.")
    svz: float | None = Field(default=None, description="Standard deviation in vz.")
    svh: float | None = Field(default=None, description="Standard deviation in vh.")

    @field_validator(
        "trg_sh", "ssd", "shd", "svd", "sdx", "sdy", "sdz", "sa", "shz", "svz", "svh"
    )
    def check_sigma(cls, v):
        return _validator(v)

    @field_validator("sd", "hd")
    def check_distance(cls, v):
        return _validator(v, error_message="Distance values must be >= 0.")

    COLUMN_LABELS: ClassVar[dict] = {
        "station_key": ["stn_pk"],
        "points_label": ["stn_id", "trg_id"],
        "points_height": ["stn_h", "trg_h"],
        "points_height_sigma": ["stn_sh", "trg_sh"],
        "linear_measurements": ["sd", "hd", "vd", "dx", "dy", "dz"],
        "linear_measurements_sigma": ["ssd", "shd", "svd", "sdx", "sdy", "sdz"],
        "angular_measurements": ["a", "hz", "vz", "vh"],
        "angular_measurements_sigma": ["sa", "shz", "svz", "svh"],
    }


class ControlPointModel(BaseModel):
    id: str
    x: float | None = None
    y: float | None = None
    z: float | None = None
    sx: float | None = Field(default=None, description="Standard deviation in x.")
    sy: float | None = Field(default=None, description="Standard deviation in y.")
    sz: float | None = Field(default=None, description="Standard deviation in z.")

    @field_validator("sx", "sy", "sz")
    def check_sigma(cls, v):
        _validator(v, enable_minus_one=True)

    COLUMN_LABELS: ClassVar[dict] = {
        "point_label": ["id"],
        "coordinates": ["x", "y", "z"],
        "sigma": ["sx", "sy", "sz"],
    }


class StationModel(BaseModel):
    stn_pk: int
    stn_id: str
    stn_h: float | None = None
    stn_sh: float | None = Field(
        default=None, description="Standard deviation in stn_h."
    )
    orientation: float | None = None

    COLUMN_LABELS: ClassVar[dict] = {
        "station_key": ["stn_pk"],
        "base_point": ["stn_id"],
        "station_attributes": ["stn_h", "stn_sh", "orientation"],
    }

    @field_validator("stn_sh")
    def check_sigma(cls, v):
        return _validator(v)
