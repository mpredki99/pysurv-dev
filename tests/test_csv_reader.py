import os
import tempfile

import pandas as pd
import pytest

from pysurv.exceptions.exceptions import (
    EmptyDatasetError,
    InvalidDataError,
    MissingMandatoryColumnsError,
)
from pysurv.reader.csv_reader import CSVReader

# Test data
VALID_MEASUREMENTS_DATA = {
    "stn_id": ["STN1", "STN1", "STN2"],
    "stn_h": [1.500, -1.576, 0.000],
    "trg_id": ["T1", "T2", "T3"],
    "trg_h": [-1.245, None, 1.753],
    "sd": [100.0, 200.0, 300.0],
    "hz": [100.0000, 134.5678, 321.1234],
    "shz": [0.0020, 0.0000, None],
    "vd": [2.0, -3.0, 14.0],
    "unnecesary": ["A", "B", "C"],
}

VALID_CONTROLS_DATA = {
    "id": ["C1", "C2", "C3"],
    "x": [1000.00, 2000.00, 3000.00],
    "y": [1000.00, 2000.00, 3000.00],
    "z": [100.00, 200.00, 300.00],
    "sx": [-1, 0.00, 1.00],
    "sy": [0.10, 0.10, 0.10],
    "sz": [0.00, 0.00, 0.00],
    "unnecesary": ["A", "B", "C"],
}

INVALID_MEASUREMENTS_DATA = {
    "stn_id": ["STN1", "STN1", "STN2"],
    "stn_h": [None, -1.576, 0],
    "trg_id": ["T1", "T2", "T3"],
    "trg_h": [-1.245, 0.000, 1.753],
    "sd": ["Invalid_value", 200.0, 300.0],
    "hz": [100.0000, 134.5678, 321.1234],
    "shz": [-0.0020, 0.0000, None],
    "vd": ["Invalid_value", 2.0, "Invalid_value"],
}

INVALID_CONTROLS_DATA = {
    "id": ["C1", "C2", "C3"],
    "x": ["Invalid_value", 2000.0, 3000.0],
    "y": [1000.0, 2000.0, 3000.0],
    "z": [100.0, "Invalid_value", 300.00],
    "sx": [-0.10, 0.00, 0.10],
    "sy": ["Invalid_value", "Invalid_value", 0.10],
    "sz": [0.00, 0.00, 0.00],
}


@pytest.fixture
def valid_controls_file():
    with tempfile.TemporaryDirectory() as temp_dir:
        valid_controls_file = os.path.join(temp_dir, "measurements.csv")
        pd.DataFrame(VALID_CONTROLS_DATA).to_csv(valid_controls_file, index=False)
        yield valid_controls_file


@pytest.fixture
def valid_measurements_file():
    with tempfile.TemporaryDirectory() as temp_dir:
        valid_measurements_file = os.path.join(temp_dir, "measurements.csv")
        pd.DataFrame(VALID_MEASUREMENTS_DATA).to_csv(
            valid_measurements_file, index=False
        )
        yield valid_measurements_file


@pytest.fixture
def invalid_measurements_file():
    with tempfile.TemporaryDirectory() as temp_dir:
        invalid_measurements_file = os.path.join(temp_dir, "invalid_measurements.csv")
        pd.DataFrame(INVALID_MEASUREMENTS_DATA).to_csv(
            invalid_measurements_file, index=False
        )
        yield invalid_measurements_file


@pytest.fixture
def invalid_controls_file():
    with tempfile.TemporaryDirectory() as temp_dir:
        invalid_controls_file = os.path.join(temp_dir, "invalid_controls.csv")
        pd.DataFrame(INVALID_CONTROLS_DATA).to_csv(invalid_controls_file, index=False)
        yield invalid_controls_file


@pytest.fixture
def empty_file():
    with tempfile.TemporaryDirectory() as temp_dir:
        empty_file = os.path.join(temp_dir, "empty.csv")
        pd.DataFrame({"id": [], "stn_id": [], "trg_id": []}).to_csv(
            empty_file, index=False
        )
        yield empty_file


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


class TestCSVReader:

    def test_validation_mode(self, valid_measurements_file, valid_controls_file):
        raise_mode = CSVReader(valid_measurements_file, valid_controls_file)
        assert raise_mode._validation_mode == "raise"

        skip_mode = CSVReader(
            valid_measurements_file, valid_controls_file, validation_mode="skip"
        )
        assert skip_mode._validation_mode == "skip"

        none_mode = CSVReader(
            valid_measurements_file, valid_controls_file, validation_mode=None
        )
        assert none_mode._validation_mode is None

        with pytest.raises(ValueError):
            CSVReader(
                valid_measurements_file,
                valid_controls_file,
                validation_mode="invalid_mode",
            )

    def test_invalid_measurement_path(self, valid_controls_file):
        with pytest.raises(FileNotFoundError) as e_info:
            CSVReader("invalid/path/measurements.csv", valid_controls_file)
        assert "Measurements file not found:" in str(e_info.value)

    def test_invalid_controls_path(self, valid_measurements_file):
        with pytest.raises(FileNotFoundError) as e_info:
            reader = CSVReader(valid_measurements_file, "invalid/path/controls.csv")
        assert "Controls file not found:" in str(e_info.value)

    def test_column_name_standarization(self, temp_dir):
        measurements_file_path = os.path.join(temp_dir, "no_columns_measurements.csv")
        pd.DataFrame(VALID_MEASUREMENTS_DATA).to_csv(
            measurements_file_path,
            index=False,
            header=[
                "STN_ID",
                "STN_H",
                "TRG_ID",
                "TRG_H",
                "SD",
                "HZ",
                "SHZ",
                "VD",
                "UNNECESARY",
            ],
        )

        controls_file_path = os.path.join(temp_dir, "no_columns_controls.csv")
        pd.DataFrame(VALID_CONTROLS_DATA).to_csv(
            controls_file_path,
            index=False,
            header=[
                "NR",
                "EASTING",
                "NORTHING",
                "ELEVATION",
                "SE",
                "SN",
                "SEL",
                "UNNECESARY",
            ],
        )

        reader = CSVReader(measurements_file_path, controls_file_path)
        reader.read_measurements()
        reader.read_controls()

        assert "stn_pk" in reader._measurements.columns
        assert "trg_id" in reader._measurements.columns
        assert "trg_h" in reader._measurements.columns
        assert "sd" in reader._measurements.columns
        assert "hz" in reader._measurements.columns
        assert "shz" in reader._measurements.columns
        assert "vd" in reader._measurements.columns

        assert "id" in reader._controls.columns
        assert "x" in reader._controls.columns
        assert "y" in reader._controls.columns
        assert "z" in reader._controls.columns
        assert "sx" in reader._controls.columns
        assert "sy" in reader._controls.columns
        assert "sz" in reader._controls.columns

    def test_validate_mandatory_columns(self, temp_dir):
        measurements_file = pd.DataFrame(VALID_MEASUREMENTS_DATA)
        measurements_file = measurements_file.drop(columns=["stn_id", "trg_id"])

        controls_file = pd.DataFrame(VALID_CONTROLS_DATA)
        controls_file = controls_file.drop(columns="id")

        measurements_file_path = os.path.join(temp_dir, "no_columns_measurements.csv")
        pd.DataFrame(measurements_file).to_csv(measurements_file_path, index=False)

        controls_file_path = os.path.join(temp_dir, "no_columns_controls.csv")
        pd.DataFrame(controls_file).to_csv(controls_file_path, index=False)

        reader = CSVReader(measurements_file_path, controls_file_path)

        with pytest.raises(MissingMandatoryColumnsError):
            reader.read_measurements()

        with pytest.raises(MissingMandatoryColumnsError):
            reader.read_controls()

    def test_filter_columns(self, valid_measurements_file, valid_controls_file):
        reader = CSVReader(valid_measurements_file, valid_controls_file)
        reader.read_measurements()
        reader.read_controls()
        assert "unnecesary" not in reader._measurements.columns
        assert "unnecesary" not in reader._controls.columns

    def test_measurements_data_validation(
        self, invalid_measurements_file, valid_controls_file, empty_file
    ):
        none_reader = CSVReader(
            invalid_measurements_file, valid_controls_file, validation_mode=None
        )
        none_reader.read_measurements()
        assert not none_reader.measurements.empty
        assert len(none_reader.measurements) == 3

        skip_reader = CSVReader(
            invalid_measurements_file, valid_controls_file, validation_mode="skip"
        )
        skip_reader.read_measurements()
        assert not skip_reader.measurements.empty
        assert skip_reader.measurements.isna().sum().sum() == 5

        raise_reader = CSVReader(invalid_measurements_file, valid_controls_file)
        with pytest.raises(InvalidDataError):
            raise_reader.read_measurements()

        for validation_mode in ["skip", "raise"]:
            empty_reader = CSVReader(
                empty_file, valid_controls_file, validation_mode=validation_mode
            )
            with pytest.raises(EmptyDatasetError):
                empty_reader.read_measurements()

    def test_controls_data_validation(
        self, valid_measurements_file, invalid_controls_file, empty_file
    ):
        none_reader = CSVReader(
            valid_measurements_file, invalid_controls_file, validation_mode=None
        )
        none_reader.read_controls()
        assert not none_reader.controls.empty
        assert len(none_reader.controls) == 3

        skip_reader = CSVReader(
            valid_measurements_file, invalid_controls_file, validation_mode="skip"
        )
        skip_reader.read_controls()
        assert not skip_reader.controls.empty
        assert skip_reader.controls.isna().sum().sum() == 5

        raise_reader = CSVReader(valid_measurements_file, invalid_controls_file)
        with pytest.raises(InvalidDataError):
            raise_reader.read_controls()

        for validation_mode in ["skip", "raise"]:
            empty_reader = CSVReader(
                valid_measurements_file, empty_file, validation_mode=validation_mode
            )
            with pytest.raises(EmptyDatasetError):
                empty_reader.read_controls()
