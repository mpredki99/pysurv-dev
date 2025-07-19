from abc import abstractmethod
from typing import Iterable

import pandas as pd

from pysurv.validators._validators import validate_angle_unit


class AngularDataset(pd.DataFrame):
    def __init__(
        self,
        data: Iterable | dict | pd.DataFrame,
        angle_unit: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(data, **kwargs)
        self._angle_unit = validate_angle_unit(angle_unit)

    @property
    def angle_unit(self) -> str:
        """Return the angle unit used for angular measurements displaying."""
        return self._angle_unit

    @angle_unit.setter
    def angle_unit(self, new_angle_unit: str | None) -> None:
        """Set the angle unit for angular measurements displaying."""
        self._angle_unit = validate_angle_unit(new_angle_unit)

    def to_dataframe(self) -> pd.DataFrame:
        """Return a pandas DataFrame representation of the dataset."""
        return pd.DataFrame(self)

    @abstractmethod
    def display(self, new_angle_unit: str | None = None):
        """Abstract method for displaying dataset with angles conversion."""
        pass
