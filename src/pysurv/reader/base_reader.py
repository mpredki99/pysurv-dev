# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

from abc import ABC, abstractmethod
from pprint import pformat
from warnings import warn

from pydantic import ValidationError

from pysurv.exceptions._exceptions import (
    EmptyDatasetError,
    InvalidDataError,
    MissingMandatoryColumnsError,
)
from pysurv.validators._models import ControlPointModel, MeasurementModel, StationModel
from pysurv.warnings._warnings import InvalidDataWarning


class BaseReader(ABC):
    """
    Abstract base class for dataset readers in PySurv.

    This class defines the interface for reading control points, measurements, and stations
    from various data sources. Subclasses must implement the read_measurements, read_controls
    and read_stations abstract methods to provide specific reading logic.
    """

    def __init__(self, validation_mode: str | None = "raise") -> None:
        self._validation_mode = self._validate_mode(validation_mode)

        self._measurements = None
        self._controls = None
        self._stations = None

    def _validate_mode(self, validation_mode: str | None):
        """Validate the validation_mode."""
        if not validation_mode in [None, "raise", "skip"]:
            raise ValueError(f"validation_mode must be either None, 'raise' or 'skip'")
        return validation_mode

    @abstractmethod
    def read_controls(self):
        """Abstract method to read control points data."""
        pass

    @abstractmethod
    def read_measurements(self):
        """Abstract method to read measurements data."""
        pass

    @abstractmethod
    def read_stations(self):
        """Abstract method to read stations data."""
        pass

    @property
    def measurements(self):
        """Return the measurements dataset."""
        return self._measurements

    @property
    def controls(self):
        """Return the controls dataset."""
        return self._controls

    @property
    def stations(self):
        """Return the stations dataset."""
        return self._stations

    def get_dataset(self, dataset_name: str):
        """Retrieve a dataset by its name."""
        datasets = {
            "Measurements": self.measurements,
            "Controls": self.controls,
            "Stations": self.stations,
        }
        if dataset_name not in datasets.keys():
            raise KeyError(f"Reader has not dataset: {dataset_name}")
        return datasets[dataset_name]

    def _filter_columns(self, dataset_name: str):
        """Filter columns of the specified dataset to only include acceptable columns."""
        acceptable_columns_dict = {
            "Measurements": MeasurementModel.COLUMN_LABELS["station_key"]
            + MeasurementModel.COLUMN_LABELS["points_label"]
            + MeasurementModel.COLUMN_LABELS["points_height"]
            + MeasurementModel.COLUMN_LABELS["points_height_sigma"]
            + MeasurementModel.COLUMN_LABELS["linear_measurements"]
            + MeasurementModel.COLUMN_LABELS["linear_measurements_sigma"]
            + MeasurementModel.COLUMN_LABELS["angular_measurements"]
            + MeasurementModel.COLUMN_LABELS["angular_measurements_sigma"],
            "Controls": ControlPointModel.COLUMN_LABELS["point_label"]
            + ControlPointModel.COLUMN_LABELS["coordinates"]
            + ControlPointModel.COLUMN_LABELS["sigma"],
            "Stations": StationModel.COLUMN_LABELS["station_key"]
            + StationModel.COLUMN_LABELS["base_point"]
            + StationModel.COLUMN_LABELS["station_attributes"],
        }
        acceptable_columns_list = acceptable_columns_dict.get(dataset_name)
        dataset = self.get_dataset(dataset_name)

        return dataset.loc[:, dataset.columns.isin(acceptable_columns_list)]

    def _to_float(self, dataset_name: str):
        """Convert numeric columns of the specified dataset to float type."""
        dataset = self.get_dataset(dataset_name)
        numeric_columns_dict = {
            "Measurements": MeasurementModel.COLUMN_LABELS["points_height"]
            + MeasurementModel.COLUMN_LABELS["points_height_sigma"]
            + MeasurementModel.COLUMN_LABELS["linear_measurements"]
            + MeasurementModel.COLUMN_LABELS["linear_measurements_sigma"]
            + MeasurementModel.COLUMN_LABELS["angular_measurements"]
            + MeasurementModel.COLUMN_LABELS["angular_measurements_sigma"],
            "Controls": ControlPointModel.COLUMN_LABELS["coordinates"]
            + ControlPointModel.COLUMN_LABELS["sigma"],
            "Stations": StationModel.COLUMN_LABELS["station_attributes"],
        }
        numeric_columns_list = numeric_columns_dict.get(dataset_name)
        numeric_columns = dataset.columns[dataset.columns.isin(numeric_columns_list)]

        for col in numeric_columns:
            dataset[col] = dataset[col].astype(float)

    def _validate_mandatory_columns(self, dataset_name: str):
        """Check that all mandatory columns are present in the specified dataset."""
        mandatory_columns_dict = {
            "Measurements": MeasurementModel.COLUMN_LABELS["points_label"],
            "Controls": ControlPointModel.COLUMN_LABELS["point_label"],
            "Stations": StationModel.COLUMN_LABELS["station_key"]
            + StationModel.COLUMN_LABELS["base_point"],
        }
        dataset = self.get_dataset(dataset_name)
        mandatory_columns_set = set(mandatory_columns_dict.get(dataset_name))
        if not mandatory_columns_set.issubset(dataset.columns):
            raise MissingMandatoryColumnsError(
                f"Missing mandatory columns in {dataset_name} dataset: {mandatory_columns_set}"
            )

    def _validate_data(self, dataset_name: str):
        """Validate the data in the specified dataset using the appropriate pydantic model."""
        models = {
            "Measurements": MeasurementModel,
            "Controls": ControlPointModel,
            "Stations": StationModel,
        }
        errors = {}
        dataset = self.get_dataset(dataset_name)
        model = models[dataset_name]

        for row_idx, row in enumerate(dataset.itertuples(index=False)):
            try:
                model(**row._asdict())
            except ValidationError as e:
                errors.update({row_idx: e})

                if self._validation_mode == "skip":
                    # Replace invalid value with None
                    cols_idx = [
                        dataset.columns.get_loc(error.get("loc")[0])
                        for error in e.errors()
                    ]
                    dataset.iloc[row_idx, cols_idx] = None

        validation_result_message = (
            f"Validation errors in {dataset_name} dataset:" + "\n"
        )
        validation_result_message += pformat(errors, indent=0)

        if errors and self._validation_mode == "raise":
            raise InvalidDataError(validation_result_message)

        elif errors and self._validation_mode == "skip":
            validation_result_message += "\n" + "Invalid values were skipped."
            warn(validation_result_message, category=InvalidDataWarning)

        dataset.dropna(axis=0, how="all", inplace=True)
        dataset.dropna(axis=1, how="all", inplace=True)

        if dataset.empty:
            raise EmptyDatasetError(f"{dataset_name} dataset is empty.")
