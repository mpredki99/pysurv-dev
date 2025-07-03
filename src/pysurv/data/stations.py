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

    def approx_oreintation_constant(self, hz, controls):
        stn_pk_loc = hz.index.names.index("stn_pk")
        trg_id_loc = hz.index.names.index("trg_id")

        for idx, hz_value in hz.items():
            stn_pk = idx[stn_pk_loc]
            stn_id = self.at[stn_pk, "stn_id"]
            trg_id = idx[trg_id_loc]

            x_first, y_first = controls.at[stn_id, "x"], controls.at[stn_id, "y"]
            x_second, y_second = controls.at[trg_id, "x"], controls.at[trg_id, "y"]
            self.at[stn_pk, "orientation"] = (
                azimuth(x_first, y_first, x_second, y_second) - hz_value
            )

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
