from typing import ClassVar

from pydantic import BaseModel, Field, field_validator


class Measurement(BaseModel):
    stn_iloc: int
    trg_id: str
    trg_h: float | None = None
    sd: float | None = Field(default=None, ge=0)
    hd: float | None = Field(default=None, ge=0)
    vd: float | None = None
    dx: float | None = None
    dy: float | None = None
    dz: float | None = None
    ssd: float | None = Field(default=None, ge=0)
    shd: float | None = Field(default=None, ge=0)
    svd: float | None = Field(default=None, ge=0)
    sdx: float | None = Field(default=None, ge=0)
    sdy: float | None = Field(default=None, ge=0)
    sdz: float | None = Field(default=None, ge=0)
    a: float | None = None
    hz: float | None = None
    vz: float | None = None
    vh: float | None = None
    sa: float | None = Field(default=None, ge=0)
    shz: float | None = Field(default=None, ge=0)
    svz: float | None = Field(default=None, ge=0)
    svh: float | None = Field(default=None, ge=0)

    COLUMN_LABELS: ClassVar[dict] = {
        "points": ["stn_id", "trg_id"],
        "points_height": ["stn_h", "trg_h"],
        "linear_measurements": ["sd", "hd", "vd", "dx", "dy", "dz"],
        "linear_measurements_sigma": ["ssd", "shd", "svd", "sdx", "sdy", "sdz"],
        "angular_measurements": ["a", "hz", "vz", "vh"],
        "angular_measurements_sigma": ["sa", "shz", "svz", "svh"],
    }


class ControlPoint(BaseModel):
    id: str
    x: float | None = None
    y: float | None = None
    z: float | None = None
    sx: float | None = Field(default=None, description="Standard deviation in x.")
    sy: float | None = Field(default=None, description="Standard deviation in y.")
    sz: float | None = Field(default=None, description="Standard deviation in z.")

    @field_validator("sx", "sy", "sz")
    def check_sigma(cls, v):
        if v is not None and not (v >= 0 or v == -1):
            raise ValueError("Sigma values must be >= 0 or -1.")
        return v

    COLUMN_LABELS: ClassVar[dict] = {
        "point_label": ["id"],
        "coordinates": ["x", "y", "z"],
        "sigma": ["sx", "sy", "sz"],
    }


class Station(BaseModel):
    stn_pk: int
    stn_id: str
    stn_h: float | None = None

    COLUMN_LABELS: ClassVar[dict] = {
        "station_key": ["stn_pk"],
        "point_label": ["stn_id"],
        "station_height": ["stn_h"],
    }
