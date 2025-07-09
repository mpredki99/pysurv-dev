from abc import ABC, abstractmethod
from pprint import pformat

from pydantic import ValidationError

from pysurv.exceptions import (
    EmptyDatasetError,
    InvalidDataError,
    MissingMandatoryColumnsError,
)
from pysurv.models.models import ControlPointModel, MeasurementModel, StationModel


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
    @property
    def measurements(self):
        return self._measurements

    @property
    def controls(self):
        return self._controls

    @property
    def stations(self):
        return self._stations

    def get_dataset(self, dataset_name):
        datasets = {
            "Measurements": self.measurements,
            "Controls": self.controls,
            "Stations": self.stations,
        }
        dataset = datasets.get(dataset_name)
        return dataset

    # Datasets filter
    def _filter_columns(self, dataset_name):
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

    def _to_float(self, dataset_name):
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

    # Datasets validators
    def _validate_mandatory_columns(self, dataset_name):
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

    def _validate_data(self, dataset_name):
        models = {
            "Measurements": MeasurementModel,
            "Controls": ControlPointModel,
            "Stations": StationModel,
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
            print(message + "\n" + "Invalid values  were skipped.")
        
        dataset.dropna(axis=0, how='all', inplace=True)
        dataset.dropna(axis=1, how='all', inplace=True)
        
        if dataset is None or dataset.empty:
            raise EmptyDatasetError(f"{dataset_name} dataset is empty.")
