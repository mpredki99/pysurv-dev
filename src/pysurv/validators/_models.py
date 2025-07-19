# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

from typing import ClassVar

from pydantic import BaseModel, Field, field_validator

from ._validators import sigma_validator


class MeasurementModel(BaseModel):
    """
    Model representing a measurement record in a measurements dataset.

    This model is used to validate information about a single measurement row,
    including station and target identifiers, measured values (distances, angles, coordinate differences),
    and their associated standard deviations.
    """
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
    def validate_sigma(cls, v):
        """Validate sigma fields for non-negative values."""
        return sigma_validator(v)

    @field_validator("sd", "hd")
    def validate_distance(cls, v):
        """Validate that distance values are non-negative."""
        return sigma_validator(v, error_message="Distance values must be >= 0.")

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
    """
    Model representing a control point in controls dataset.

    This model is used to validate information about a control point,
    including its identifier, coordinates, and associated standard deviations.
    """
    id: str
    x: float | None = None
    y: float | None = None
    z: float | None = None
    sx: float | None = Field(default=None, description="Standard deviation in x.")
    sy: float | None = Field(default=None, description="Standard deviation in y.")
    sz: float | None = Field(default=None, description="Standard deviation in z.")

    @field_validator("sx", "sy", "sz")
    def validate_sigma(cls, v):
        """Validate sigma fields for non-negative values enabling special value -1."""
        sigma_validator(v, enable_minus_one=True)

    COLUMN_LABELS: ClassVar[dict] = {
        "point_label": ["id"],
        "coordinates": ["x", "y", "z"],
        "sigma": ["sx", "sy", "sz"],
    }


class StationModel(BaseModel):
    """
    Model representing a station record in stations dataset.

    This model is used to validate information about a station,
    including its primary key, identifier, height, standard deviation of height,
    and orientation.
    """
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
    def validate_sigma(cls, v):
        """Validate stn_sh field for non-negative values."""
        return sigma_validator(v)
