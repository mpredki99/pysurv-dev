import os

import numpy as np
import pandas as pd

from .base_reader import BaseReader


class CSVReader(BaseReader):

    def __init__(
        self, measurements_file_path, controls_file_path, validation_mode="raise"
    ):
        super().__init__(validation_mode)

        self.validate_file_path(measurements_file_path, "Measurements")
        self._measurements_file_path = measurements_file_path

        self.validate_file_path(controls_file_path, "Controls")
        self._controls_file_path = controls_file_path

    def validate_file_path(self, file_path, dataset_name):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{dataset_name} file not found: {file_path}")

    # MEASUREMENTS DATASET METHODS
    def read_measurements(self):
        self._measurements = pd.read_csv(self._measurements_file_path)

        self._measurements.columns = self._measurements.columns.str.lower()

        self._validate_mandatory_columns("Measurements")
        self._filter_columns("Measurements")

        self.read_stations()

        self._fill_empty_stn_labels()

        if self._validation_mode in ["raise", "skip"]:
            self._validate_data("Measurements")

    def _fill_empty_stn_labels(self):
        if "stn_h" in self._stations.columns:
            self._measurements.fillna({"stn_h": 0}, inplace=True)
            self._measurements = self._measurements.merge(self._stations, left_on=["stn_id", "stn_h"], right_on=["stn_id", "stn_h"], how="left")
        else:
            self._measurements = self._measurements.merge(self._stations, left_on="stn_id", right_on="stn_id", how="left")
        
        self._measurements.loc[: "stn_iloc"] = self._measurements.loc[: "stn_iloc"].ffill()
        self._measurements["stn_iloc"] = self._measurements["stn_iloc"].astype(int)
        self._measurements.drop(
            columns=["stn_id", "stn_h"], errors="ignore", inplace=True
        )
        stn_iloc = self._measurements.pop("stn_iloc") 
        self._measurements.insert(0, "stn_iloc", stn_iloc)

    # CONTROLS DATASET METHODS
    def read_controls(self):
        self._controls = pd.read_csv(self._controls_file_path)
        self._controls.columns = self._controls.columns.str.lower()
        self._standardize_control_columns_names()
        self._validate_mandatory_columns("Controls")
        self._controls = self._filter_columns("Controls")

        if self._validation_mode in ["raise", "skip"]:
            self._validate_data("Controls")

    def _standardize_control_columns_names(self):
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

    # STATIONS DATASET METHODS
    def read_stations(self):
        if "stn_h" not in self._measurements.columns:
            self._stations = self._measurements["stn_id"].dropna().to_frame()
        else:
            self._stations = (
                self._measurements[["stn_id", "stn_h"]].dropna(how="all").fillna({"stn_h": 0})
            )

        self._standardize_stations_columns_names()
        self._validate_mandatory_columns("Stations")
        self._filter_columns("Stations")

        if self._validation_mode in ["raise", "skip"]:
            self._validate_data("Stations")

    def _standardize_stations_columns_names(self):
        self._stations.rename_axis("stn_iloc", inplace=True)
        self._stations.reset_index(inplace=True)
