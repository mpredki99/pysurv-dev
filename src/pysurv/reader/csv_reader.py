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
        self._measurements = self._filter_columns("Measurements")

        self.read_stations()

        self._insert_stn_pk()

        if self._validation_mode in ["raise", "skip"]:
            self._validate_data("Measurements")

    def _insert_stn_pk(self):
        self._measurements.insert(0, "meas_iloc", np.arange(len(self._measurements)))
        self._measurements = self._measurements.merge(
            self._stations,
            left_on="meas_iloc",
            right_on="stn_pk",
            how="left",
            suffixes=["_measurements", "_stations"],
        )

        self._measurements.loc[:"stn_pk"] = self._measurements.loc[:"stn_pk"].ffill()
        self._measurements.drop(
            columns=[
                "stn_id_measurements",
                "stn_id_stations",
                "stn_h_measurements",
                "stn_h_stations",
                "meas_iloc",
            ],
            errors="ignore",
            inplace=True,
        )
        stn_pk = self._measurements.pop("stn_pk")
        self._measurements.insert(0, "stn_pk", stn_pk.astype(int))

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
                self._measurements[["stn_id", "stn_h"]]
                .dropna(how="all")
                .fillna({"stn_h": 0})
            )

        self._standardize_stations_columns_names()
        self._validate_mandatory_columns("Stations")
        self._filter_columns("Stations")

        if self._validation_mode in ["raise", "skip"]:
            self._validate_data("Stations")

    def _standardize_stations_columns_names(self):
        self._stations.rename_axis("stn_pk", inplace=True)
        self._stations.reset_index(inplace=True)
