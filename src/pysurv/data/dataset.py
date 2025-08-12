# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

import os
from typing import Any

import pandas as pd

from pysurv.reader.csv_reader import CSVReader

from .controls import Controls
from .measurements import Measurements
from .stations import Stations


class Dataset:
    """
    Container class for measurements, control points, and stations datasets.

    This class provides a unified interface for accessing and manipulating
    datasets, including measurements, controls, and stations. It supports
    loading data from CSV files and provides convenient views of joined
    datasets.
    """

    def __init__(
        self, measurements: Measurements, controls: Controls, stations: Stations
    ) -> None:
        self._measurements = measurements
        self._controls = controls
        self._stations = stations

    @property
    def measurements(self) -> Measurements:
        """Return the measurements dataset."""
        return self._measurements

    @property
    def measurements_view(self) -> pd.DataFrame:
        """Return a view of the measurements dataset joined with stations."""
        return self._join_measurements_with_stations(self._measurements.display())

    def _join_measurements_with_stations(
        self, measurements_dataset: Measurements
    ) -> pd.DataFrame:
        """Join measurements dataset with stations data."""
        joined_view = measurements_dataset.join(
            self._stations.display(), on="stn_pk", how="left", sort=True
        )
        joined_view.set_index(
            self._stations.columns.tolist(), append=True, inplace=True
        )

        index_names_sorted = [
            index_name
            for index_name in [
                "stn_pk",
                "stn_id",
                "stn_h",
                "stn_sh",
                "orientation",
                "trg_id",
                "trg_h",
                "trg_sh",
            ]
            if index_name in joined_view.index.names
        ]

        return joined_view.reorder_levels(index_names_sorted, axis=0)

    @property
    def controls(self) -> Controls:
        """Return the controls dataset."""
        return self._controls

    @property
    def stations(self) -> Stations:
        """Return the stations dataset."""
        return self._stations

    @classmethod
    def from_csv(
        cls,
        measurements_file_path: str | os.PathLike,
        controls_file_path: str | os.PathLike,
        validation_mode: str | None = "raise",
        angle_unit: str | None = None,
        swap_xy: bool = False,
        delimiter: str | None = None,
        decimal: str = ".",
        crs: Any = None,
    ):
        """Create a Dataset instance from CSV files."""
        reader = CSVReader(
            measurements_file_path,
            controls_file_path,
            validation_mode,
            delimiter=delimiter,
            decimal=decimal,
        )
        reader.read_measurements()
        reader.read_controls()

        measurements = Measurements(reader.measurements, angle_unit=angle_unit)
        controls = Controls(reader.controls, swap_xy=swap_xy, crs=crs)
        stations = Stations(reader.stations)

        return cls(measurements, controls, stations)
