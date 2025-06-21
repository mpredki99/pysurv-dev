import pandas as pd

from pysurv.basic import from_rad, to_rad
from pysurv.models import MeasurementModel


class Measurements(pd.DataFrame):
    def __init__(self, data, *args, angle_unit="grad", **kwargs):
        super().__init__(data, *args, **kwargs)
        self._validate_angle_unit(angle_unit)
        self._angle_unit = angle_unit

        if {"stn_pk", "trg_id"}.issubset(self.columns):
            self.set_index(["stn_pk", "trg_id"], inplace=True)
            self._to_float()
            self.angles_to_rad()

    def _validate_angle_unit(self, angle_unit):
        if angle_unit not in ["rad", "grad", "gon", "deg"]:
            raise ValueError("Angle unit must be either 'rad', 'grad', 'gon', 'deg'.")

    def _to_float(self):
        for col in self.columns:
            self[col] = self[col].astype(float)

    @property
    def angular_measurements_columns(self):
        return self.columns[
            self.columns.isin(MeasurementModel.COLUMN_LABELS["angular_measurements"])
        ]

    @property
    def angular_sigma_columns(self):
        return self.columns[
            self.columns.isin(
                MeasurementModel.COLUMN_LABELS["angular_measurements_sigma"]
            )
        ]

    @property
    def angular_columns(self):
        return self.angular_measurements_columns.append(self.angular_sigma_columns)

    @property
    def linear_measurements_columns(self):
        return self.columns[
            self.columns.isin(MeasurementModel.COLUMN_LABELS["linear_measurements"])
        ]

    @property
    def linear_sigma_columns(self):
        return self.columns[
            self.columns.isin(
                MeasurementModel.COLUMN_LABELS["linear_measurements_sigma"]
            )
        ]

    @property
    def linear_columns(self):
        return self.linear_measurements_columns.append(self.linear_sigma_columns)

    @property
    def angle_unit(self):
        return self._angle_unit

    @property
    def _constructor(self):
        return Measurements

    def set_angle_unit(self, new_angle_unit):
        self._validate_angle_unit(new_angle_unit)
        self._angle_unit = new_angle_unit

    def angles_to_rad(self):
        if self.angle_unit == "rad":
            return
        self[self.angular_columns] = to_rad(
            self[self.angular_columns], unit=self.angle_unit
        )

    def display(self):
        measurements = self.copy()
        if self.angle_unit != "rad":
            measurements[self.angular_columns] = from_rad(
                measurements[self.angular_columns], unit=self.angle_unit
            )
        return measurements
