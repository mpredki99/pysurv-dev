import pandas as pd

from pysurv.models import StationModel


class Stations(pd.DataFrame):
    def __init__(self, data, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        if {"stn_pk"}.issubset(self.columns):
            self.set_index("stn_pk", inplace=True)
            self._to_float()

    def _to_float(self):
        cols = (
            StationModel.COLUMN_LABELS["station_height"]
            + StationModel.COLUMN_LABELS["station_height_sigma"]
        )
        for col in self.columns[self.columns.isin(cols)]:
            self[col] = self[col].astype(float)

    @property
    def _constructor(self):
        return Stations
