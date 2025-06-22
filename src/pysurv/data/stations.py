import pandas as pd


class Stations(pd.DataFrame):
    def __init__(self, data, *args, **kwargs):
        super().__init__(data, *args, **kwargs)

        if {"stn_pk"}.issubset(self.columns):
            self.set_index("stn_pk", inplace=True)

    @property
    def _constructor(self):
        return Stations
