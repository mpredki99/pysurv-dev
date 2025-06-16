import os

from pysurv.reader.csv_reader import CSVReader


class Dataset:
    def __init__(self, measurements, controls, stations):
        self._measurements = measurements
        self._controls = controls
        self._stations = stations

    @classmethod
    def from_csv(
        cls, measurements_file_path, controls_file_path, validation_mode="raise"
    ):

        reader = CSVReader(measurements_file_path, controls_file_path, validation_mode)
        reader.read_measurements()
        reader.read_controls()

        return cls(
            reader.get_measurements(), reader.get_controls(), reader.get_stations()
        )
