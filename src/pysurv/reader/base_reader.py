from abc import ABC, abstractmethod
from pprint import pformat

from pydantic import ValidationError
import numpy as np

from pysurv.models.models import ControlPoint, Measurement, Station
from pysurv.exceptions.exceptions import (
    EmptyDatasetError,
    InvalidDataError,
    MissingMandatoryColumnsError,
)


class BaseReader(ABC):

    def __init__(self, validation_mode="raise"):

        self._validate_mode(validation_mode)
        self._validation_mode = validation_mode

        self._measurements = None
        self._controls = None
        self._stations = None

    def _validate_mode(self, validation_mode):
        if not validation_mode in [None, "raise", "skip"]:
            raise ValueError(f"validation_mode must be either None, 'raise' or 'skip'")

    # Abstract dataset reader methods
    @abstractmethod
    def read_controls(self):
        pass

    @abstractmethod
    def read_measurements(self):
        pass

    @abstractmethod
    def read_stations(self):
        pass

    # Dataset getters
    def get_measurements(self):
        return self._measurements

    def get_controls(self):
        return self._controls

    def get_stations(self):
        return self._stations

    def get_dataset(self, dataset_name):
        dataset_getters = {
            "Measurements": self.get_measurements,
            "Controls": self.get_controls,
            "Stations": self.get_stations,
        }
        dataset_getter = dataset_getters.get(dataset_name)
        return dataset_getter()

    # Datasets filter
    def _filter_columns(self, dataset_name):
        acceptable_columns_dict = {
            "Measurements": Measurement.COLUMN_LABELS["station_key"]
            + Measurement.COLUMN_LABELS["points"]
            + Measurement.COLUMN_LABELS["points_height"]
            + Measurement.COLUMN_LABELS["points_height_sigma"]
            + Measurement.COLUMN_LABELS["linear_measurements"]
            + Measurement.COLUMN_LABELS["linear_measurements_sigma"]
            + Measurement.COLUMN_LABELS["angular_measurements"]
            + Measurement.COLUMN_LABELS["angular_measurements_sigma"],
            "Controls": ControlPoint.COLUMN_LABELS["point_label"]
            + ControlPoint.COLUMN_LABELS["coordinates"]
            + ControlPoint.COLUMN_LABELS["sigma"],
            "Stations": Station.COLUMN_LABELS["station_key"]
            + Station.COLUMN_LABELS["point_label"]
            + Station.COLUMN_LABELS["station_height"],
        }
        acceptable_columns_list = acceptable_columns_dict.get(dataset_name)
        dataset = self.get_dataset(dataset_name)

        return dataset.loc[:, dataset.columns.isin(acceptable_columns_list)]

    # Datasets validators
    def _validate_mandatory_columns(self, dataset_name):
        mandatory_columns_dict = {
            "Measurements": Measurement.COLUMN_LABELS["points"],
            "Controls": ControlPoint.COLUMN_LABELS["point_label"],
            "Stations": Station.COLUMN_LABELS["station_key"]
            + Station.COLUMN_LABELS["point_label"],
        }
        dataset = self.get_dataset(dataset_name)
        mandatory_columns_set = set(mandatory_columns_dict.get(dataset_name))
        if not mandatory_columns_set.issubset(dataset.columns):
            raise MissingMandatoryColumnsError(
                f"Missing mandatory columns in {dataset_name} dataset: {mandatory_columns_set}"
            )

    def _validate_data(self, dataset_name):
        models = {
            "Measurements": Measurement,
            "Controls": ControlPoint,
            "Stations": Station,
        }
        errors = {}
        dataset = self.get_dataset(dataset_name)
        model = models.get(dataset_name)

        for row_idx, row in enumerate(dataset.itertuples(index=False)):
            try:
                model(**row._asdict())
            except ValidationError as e:
                errors.update({row_idx: e})
                
                if self._validation_mode == "skip": 
                    col_idx = [
                        dataset.columns.get_loc(error.get("loc")[0])
                        for error in e.errors()
                    ]
                    dataset.iloc[row_idx, col_idx] = None

        message = (
            f"Validation errors in {dataset_name} dataset:\n{pformat(errors, indent=0)}"
        )
        if errors and self._validation_mode == "raise":
            raise InvalidDataError(message)
        elif errors and self._validation_mode == "skip":
            print(message + "\nInvalid values  were skipped.")
            invalid_row_indices = dataset.iloc[list(errors.keys())].index
        if dataset.empty or dataset is None:
            raise EmptyDatasetError(f"{dataset_name} dataset is empty.")
