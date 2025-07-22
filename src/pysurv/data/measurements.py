# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

from typing import Iterable

import pandas as pd

from pysurv.basic.basic import from_rad, to_rad
from pysurv.validators._models import MeasurementModel
from pysurv.validators._validators import validate_angle_unit

from .angular_dataset import AngularDataset


class Measurements(AngularDataset):
    """
    Measurements dataset class for storing and managing surveying measurements data.

    Internally sets index from columns available in ['stn_pk', 'trg_id', 'trg_h', 'trg_sh']
    and converts angles to radians.

    Inherits from
    -------------
    pandas.DataFrame
    """

    _metadata = ["_angle_unit"]

    def __init__(
        self,
        data: Iterable | dict | pd.DataFrame,
        angle_unit: str | None = None,
        **kwargs,
    ) -> None:

        _first_init = kwargs.pop("_first_init", True)
        super().__init__(data, **kwargs)

        self._angle_unit = validate_angle_unit(angle_unit)

        if _first_init:
            index_columns = [
                col
                for col in ["stn_pk", "trg_id", "trg_h", "trg_sh"]
                if col in self.columns
            ]
            self.set_index(index_columns, inplace=True)
            self._angles_to_rad()

    @property
    def _constructor(self):
        """Return class constructor with hidden _first_init kwarg."""

        def _c(*args, **kwargs):
            kwargs["_first_init"] = False
            return Measurements(*args, **kwargs)

        return _c

    def _angles_to_rad(self) -> None:
        """Convert all angular measurement columns to radians."""
        if self.angle_unit == "rad":
            return

        self[self.angular_columns] = to_rad(
            self[self.angular_columns], unit=self.angle_unit
        )

    @property
    def linear_measurement_columns(self) -> pd.Index:
        """Return columns corresponding to linear measurements."""
        return self.columns[
            self.columns.isin(MeasurementModel.COLUMN_LABELS["linear_measurements"])
        ]

    @property
    def linear_sigma_columns(self) -> pd.Index:
        """Return columns corresponding to linear measurements standard deviations."""
        return self.columns[
            self.columns.isin(
                MeasurementModel.COLUMN_LABELS["linear_measurements_sigma"]
            )
        ]

    @property
    def linear_columns(self) -> pd.Index:
        """Return columns corresponding to linear measurements and their standard deviations."""
        return self.linear_measurement_columns.append(self.linear_sigma_columns)

    @property
    def angular_measurement_columns(self) -> pd.Index:
        """Return columns corresponding to angular measurements."""
        return self.columns[
            self.columns.isin(MeasurementModel.COLUMN_LABELS["angular_measurements"])
        ]

    @property
    def angular_sigma_columns(self) -> pd.Index:
        """Return columns corresponding to angular measurements standard deviations."""
        return self.columns[
            self.columns.isin(
                MeasurementModel.COLUMN_LABELS["angular_measurements_sigma"]
            )
        ]

    @property
    def angular_columns(self) -> pd.Index:
        """Return columns corresponding to angular measurements and their standard deviations."""
        return self.angular_measurement_columns.append(self.angular_sigma_columns)

    @property
    def measurement_columns(self) -> pd.Index:
        """Return columns corresponding to all measurements."""
        return self.linear_measurement_columns.append(self.angular_measurement_columns)

    @property
    def sigma_columns(self) -> pd.Index:
        """Return columns corresponding to all standard deviations."""
        return self.linear_sigma_columns.append(self.angular_sigma_columns)

    @property
    def measurement_data(self) -> pd.Index:
        """Return data subset corresponding to all measurements (angles in radians)."""
        return self[self.measurement_columns]

    @property
    def sigma_data(self) -> pd.Index:
        """Return data subset corresponding to all standard deviations (angles in radians)."""
        return self[self.sigma_columns]

    def display(self, angle_unit: str | None = None):
        """Return a copy of the dataset with angules converted for display."""
        measurements = self.copy()
        angle_unit = angle_unit or self._angle_unit

        if self.angle_unit != "rad":
            measurements[self.angular_columns] = from_rad(
                measurements[self.angular_columns], unit=self.angle_unit
            )
        return measurements
