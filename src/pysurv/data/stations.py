# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

from typing import Iterable

import numpy as np
import pandas as pd

from pysurv.basic.basic import azimuth, from_rad

from .angular_dataset import AngularDataset
from .controls import Controls


class Stations(AngularDataset):
    """
    Measurements dataset class for storing and managing surveying station data.

    Inherits from
    -------------
    pandas.DataFrame
    """

    def __init__(
        self,
        data: Iterable | dict | pd.DataFrame,
        angle_unit: str | None = None,
        **kwargs,
    ) -> None:
        _first_init = kwargs.pop("_first_init", True)
        super().__init__(data, **kwargs)

        if _first_init:
            self.set_index("stn_pk", inplace=True)

    @property
    def _constructor(self):
        """Return class constructor with hidden _first_init kwarg."""

        def _c(*args, **kwargs):
            kwargs["_first_init"] = False
            return Stations(*args, **kwargs)

        return _c

    def append_oreintation_constant(
        self, hz_data: pd.Series, controls: Controls
    ) -> None:
        """Append orientation constant to the dataset."""
        hz_data = hz_data.reset_index()
        hz_data.insert(1, "stn_id", hz_data.stn_pk.map(self.stn_id))

        end_points = ["stn", "trg"]
        for point in end_points:
            hz_data = hz_data.merge(
                controls[["x", "y"]],
                how="left",
                left_on=f"{point}_id",
                right_on="id",
                suffixes=[f"_{pt}" for pt in end_points],
            )
        hz_data = hz_data.set_index("stn_pk")

        self["orientation"] = (
            azimuth(
                hz_data["x_stn"], hz_data["y_stn"], hz_data["x_trg"], hz_data["y_trg"]
            )
            - hz_data["hz"]
        )

    def display(self, angle_unit: str | None = None):
        """Display the stations dataset with orientation constant in the specified angle unit."""
        orientation = "orientation"
        frame = self.copy()
        angle_unit = angle_unit or self._angle_unit

        if orientation not in frame.columns or angle_unit == "rad":
            return frame

        frame[orientation] = from_rad(
            np.mod(frame[orientation], 2 * np.pi), unit=angle_unit
        )

        return frame
