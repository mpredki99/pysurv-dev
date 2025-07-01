import pandas as pd

from pysurv.basic import from_rad, to_rad
from pysurv.config import config
from pysurv.models import (
    ControlPointModel,
    MeasurementModel,
    _validator,
    validate_angle_unit,
)

__all__ = ["sigma_config"]


class SigmaRow:

    def __init__(self, angle_unit="rad", **kwargs) -> None:
        data = {}
        for key, value in kwargs.items():
            data[key] = self._validate_attr(key, value, angle_unit=angle_unit)
        self._data = pd.Series(data)

    def __setattr__(self, key, value) -> None:
        if key == "_data":
            super().__setattr__(key, value)
        else:
            self._data[key] = self._validate_attr(key, value)

    def __getattr__(self, key):
        if key == "_data":
            return super().__getattr__(key)
        return self._data.get(key)

    def __str__(self) -> str:
        return self._data.__str__()

    def _validate_attr(self, key, value, angle_unit="rad"):
        if (
            key
            in MeasurementModel.COLUMN_LABELS["linear_measurements_sigma"]
            + MeasurementModel.COLUMN_LABELS["points_height_sigma"]
        ):
            return self._validate_distance_sigma(value)
        elif key in MeasurementModel.COLUMN_LABELS["angular_measurements_sigma"]:
            return self._validate_angle_sigma(value, angle_unit=angle_unit)
        elif key in ControlPointModel.COLUMN_LABELS["sigma"]:
            return self._validate_control_point_sigma(value)
        else:
            raise AttributeError(f"Sigma do not have attribute {key}")

    def _validate_distance_sigma(self, value):
        return _validator(value)

    def _validate_angle_sigma(self, value, angle_unit="rad"):
        value = _validator(value)
        if angle_unit == "rad":
            return value
        return to_rad(value, unit=angle_unit)

    def _validate_control_point_sigma(self, value):
        return _validator(value, enable_minus_one=True)

    def set(self, key, value, angle_unit="rad"):
        self._data[key] = self._validate_attr(key, value, angle_unit=angle_unit)

    def get(self, key, angle_unit="rad"):
        if key not in self._data.keys():
            raise AttributeError(f"Sigma do not have attribute {key}")
        if (
            angle_unit != "rad"
            and key in MeasurementModel.COLUMN_LABELS["angular_measurements_sigma"]
        ):
            return from_rad(self._data.get(key), unit=angle_unit)
        return self._data.get(key)


class SigmaConfig:
    _default = dict(
        stn_sh=0.01,
        trg_sh=0.01,
        ssd=0.01,
        shd=0.01,
        svd=0.01,
        sdx=0.001,
        sdy=0.001,
        sdz=0.001,
        sa=to_rad(0.0020),
        shz=to_rad(0.0020),
        svz=to_rad(0.0020),
        svh=to_rad(0.0020),
        sx=0.01,
        sy=0.01,
        sz=0.01,
    )

    def __init__(self) -> None:
        self.restore_default()

    def __delattr__(self, key: str) -> None:
        if key == "default":
            raise ValueError("Can not delete 'default' index.")
        else:
            super().__delattr__(key)

    def __repr__(self) -> str:
        return self._dataframe

    def __str__(self) -> str:
        return self._dataframe.to_string()

    @property
    def _dataframe(self):
        data = pd.DataFrame({idx: row._data for idx, row in self.__dict__.items()})
        return data.T

    def _validate_name(self, name):
        if name is None:
            return f"index_{len(self.__dict__)}"
        elif name.strip().isidentifier():
            return name.strip()
        raise ValueError("Attribute name is not valid identifier.")

    @property
    def index(self):
        return pd.Index(self.__dict__.keys())

    def restore_default(self):
        self.__setattr__("default", SigmaRow(angle_unit="rad", **self._default))

    def append(self, name=None, angle_unit="grad", **kwargs):
        data = {}
        for key in self._default.keys():
            data[key] = kwargs.get(key, self.default.get(key, angle_unit=angle_unit))
        name = self._validate_name(name)
        self.__setattr__(name, SigmaRow(angle_unit=angle_unit, **data))

    def display(self, angle_unit=None):
        angle_unit = (
            config.angle_unit if angle_unit is None else validate_angle_unit(angle_unit)
        )
        data = self._dataframe
        if angle_unit == "rad":
            return data
        anglular_columns = MeasurementModel.COLUMN_LABELS["angular_measurements_sigma"]
        data[anglular_columns] = from_rad(data[anglular_columns], unit=angle_unit)
        return data

    def get_row(self, index, angle_unit=None):
        angle_unit = (
            config.angle_unit if angle_unit is None else validate_angle_unit(angle_unit)
        )
        if index not in self.index:
            return IndexError(f"Sigma does not have index: {index}")
        data = self.display(angle_unit=angle_unit)
        return data.loc[index]


sigma_config = SigmaConfig()
