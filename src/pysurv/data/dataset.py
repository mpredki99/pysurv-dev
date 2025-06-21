import os

from pysurv.reader.csv_reader import CSVReader

from .controls import Controls
from .measurements import Measurements
from .stations import Stations


class Dataset:
    def __init__(self, measurements, controls, stations):
        self._measurements = measurements
        self._controls = controls
        self._stations = stations

    @classmethod
    def from_csv(
        cls,
        measurements_file_path,
        controls_file_path,
        validation_mode="raise",
        angle_unit="grad",
        swap_xy=False,
        delimiter=None,
        decimal=".",
        crs=None,
    ):

        reader = CSVReader(
            measurements_file_path,
            controls_file_path,
            validation_mode,
            delimiter=delimiter,
            decimal=decimal,
        )
        reader.read_measurements()
        reader.read_controls()

        measurements = Measurements(reader.get_measurements(), angle_unit=angle_unit)
        controls = Controls(reader.get_controls(), swap_xy=swap_xy, crs=crs)
        stations = Stations(reader.get_stations())

        return cls(measurements, controls, stations)

    @property
    def measurements(self):
        return self._measurements

    @property
    def measurements_view(self):
        return self._join_measurements_with_stations(self._measurements.display())

    @property
    def measurements_view_rad(self):
        return self._join_measurements_with_stations(self._measurements.copy())

    def _join_measurements_with_stations(self, measurements_dataset):
        joined_view = measurements_dataset.join(
            self._stations, on="stn_pk", how="left", sort=True
        )
        joined_view.set_index(
            self._stations.columns.tolist(), append=True, inplace=True
        )

        index_names_sorted = [
            index_name
            for index_name in ["stn_pk", "stn_id", "stn_h", "stn_sh", "trg_id"]
            if index_name in joined_view.index.names
        ]

        return joined_view.reorder_levels(index_names_sorted, axis=0)

    @property
    def controls(self):
        return self._controls

    @property
    def stations(self):
        return self._stations
