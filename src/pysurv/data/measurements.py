import pandas as pd

from pysurv.basic import from_rad, to_rad
from pysurv.config import config
from pysurv.models import MeasurementModel, validate_angle_unit


class Measurements(pd.DataFrame):
    def __init__(self, data, *args, angle_unit=None, **kwargs):
        super().__init__(data, *args, **kwargs)
        self._angle_unit = (
            config.angle_unit if angle_unit is None else validate_angle_unit(angle_unit)
        )

        index_columns = ["stn_pk", "trg_id"]
        optional_index_columns = [
            col for col in self.columns if col in ["trg_h", "trg_sh"]
        ]
        index_columns.extend(optional_index_columns)

        if set(index_columns).issubset(self.columns):
            self.set_index(index_columns, inplace=True)
            self._angles_to_rad()

    @property
    def _constructor(self):
        return Measurements

    # Public interface
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
    def angle_unit(self):
        return self._angle_unit

    @angle_unit.setter
    def angle_unit(self, new_angle_unit):
        self._angle_unit = validate_angle_unit(new_angle_unit)

    def _angles_to_rad(self):
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
