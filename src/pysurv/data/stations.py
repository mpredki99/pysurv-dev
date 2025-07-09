import numpy as np
import pandas as pd

from pysurv.basic import azimuth, from_rad
from pysurv.config import config
from pysurv.models import validate_angle_unit


class Stations(pd.DataFrame):
    def __init__(self, data, *args, **kwargs):
        super().__init__(data, *args, **kwargs)

        if {"stn_pk"}.issubset(self.columns):
            self.set_index("stn_pk", inplace=True)

    @property
    def _constructor(self):
        return Stations

    def append_oreintation_constant(self, data, controls):
        df1 = data.copy()
        data = data.reset_index()
        data.insert(1, "stn_id", data.stn_pk.map(self.stn_id))
        
        cols = ["stn_id", "trg_id"]
        for point in cols:
            data = data.merge(controls[["x", "y"]], how='left', left_on=point, right_on="id", suffixes=[f"_{col}" for col in cols])
        data = data.set_index("stn_pk")
        
        self["orientation"] = azimuth(data["x_stn_id"], data["y_stn_id"], data["x_trg_id"], data["y_trg_id"]) - data["hz"]

    def display(self, angle_unit=None):
        col = "orientation"
        data = self.copy()

        if col not in data.columns:
            return data

        angle_unit = (
            config.angle_unit if angle_unit is None else validate_angle_unit(angle_unit)
        )
        data[col] = from_rad(np.mod(data[col], 2 * np.pi), unit=angle_unit)

        return data
