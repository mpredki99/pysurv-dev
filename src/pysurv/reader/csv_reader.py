# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

import os

import numpy as np
import pandas as pd

from .base_reader import BaseReader


class CSVReader(BaseReader):
    """
    Concrete implementation of the BaseReader abstract class for reading datasets from CSV files.

    This class provides methods to read measurements, controls, and stations data from CSV files,
    filter and validate their contents.
    """

    def __init__(
        self,
        measurements_file_path: str | os.PathLike,
        controls_file_path: str | os.PathLike,
        validation_mode: str | None = "raise",
        delimiter: str | None = None,
        decimal: str = ".",
    ):
        super().__init__(validation_mode)

        self._measurements_file_path = self._validate_file_path(
            measurements_file_path, "Measurements"
        )
        self._controls_file_path = self._validate_file_path(
            controls_file_path, "Controls"
        )

        self.delimiter = delimiter
        self.decimal = decimal

    def _validate_file_path(self, file_path, dataset_name):
        """Validate provided path to check if file exists."""
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"{dataset_name} file not found: {file_path}")
        return file_path

    def read_measurements(self):
        """Read measurements data from CSV file."""
        self._measurements = pd.read_csv(
            self._measurements_file_path, delimiter=self.delimiter, decimal=self.decimal
        )

        self._measurements.columns = self._measurements.columns.str.lower()
        self._validate_mandatory_columns("Measurements")
        self._measurements = self._filter_columns("Measurements")

        if self._stations is None:
            self.read_stations()
            self._insert_stn_pk()

        if self._validation_mode in ["raise", "skip"]:
            self._validate_data("Measurements")

        self._to_float("Measurements")

    def _insert_stn_pk(self):
        """Insert station primary key (stn_pk) into the measurements dataset."""
        self._measurements.insert(0, "meas_iloc", np.arange(len(self._measurements)))
        self._measurements = self._measurements.merge(
            self._stations,
            left_on="meas_iloc",
            right_on="stn_pk",
            how="left",
            suffixes=["_measurements", "_stations"],
        )

        self._measurements["stn_pk"] = self._measurements["stn_pk"].ffill()
        self._measurements.drop(
            columns=[
                "stn_id_measurements",
                "stn_id_stations",
                "stn_h_measurements",
                "stn_h_stations",
                "stn_sh_measurements",
                "stn_sh_stations",
                "meas_iloc",
            ],
            errors="ignore",
            inplace=True,
        )
        stn_pk = self._measurements.pop("stn_pk")
        self._measurements.insert(0, "stn_pk", stn_pk.astype(int))

    def read_controls(self):
        """Read controls data from CSV file."""
        self._controls = pd.read_csv(
            self._controls_file_path, delimiter=self.delimiter, decimal=self.decimal
        )
        self._controls.columns = self._controls.columns.str.lower()
        self._standardize_control_columns_names()
        self._validate_mandatory_columns("Controls")
        self._controls = self._filter_columns("Controls")

        if self._validation_mode in ["raise", "skip"]:
            self._validate_data("Controls")

        self._to_float("Controls")

    def _standardize_control_columns_names(self):
        """Standardize column names in the controls dataset."""
        self._controls.rename(
            columns={
                "nr": "id",
                "easting": "x",
                "e": "x",
                "se": "sx",
                "northing": "y",
                "n": "y",
                "sn": "sy",
                "elevation": "z",
                "el": "z",
                "sel": "sz",
                "height": "z",
                "h": "z",
                "sh": "sz",
            },
            inplace=True,
        )

    def read_stations(self):
        """Extract stations data from measurements dataset."""
        if self._measurements is None:
            return self.read_measurements()

        stn_columns = self._measurements.columns[
            self._measurements.columns.isin(["stn_id", "stn_h", "stn_sh"])
        ]
        self._stations = self._measurements[stn_columns].copy()
        self._stations.dropna(how="all", inplace=True)
        self._stations.fillna({"stn_h": 0}, inplace=True)
        self._stations["stn_id"] = self._stations["stn_id"].ffill()

        self._standardize_stations_columns_names()
        self._validate_mandatory_columns("Stations")
        self._filter_columns("Stations")

        if self._validation_mode in ["raise", "skip"]:
            self._validate_data("Stations")

        self._to_float("Stations")

    def _standardize_stations_columns_names(self):
        """Standardize column names in the stations dataset."""
        self._stations.rename_axis("stn_pk", inplace=True)
        self._stations.reset_index(inplace=True)
