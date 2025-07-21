# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

from typing import List, Tuple

import pandas as pd
import pytest

from pysurv.exceptions._exceptions import (
    EmptyDatasetError,
    InvalidDataError,
    MissingMandatoryColumnsError,
)
from pysurv.reader.csv_reader import CSVReader


def test_mandatory_init_arguments() -> None:
    """Test that CSVReader requires mandatory init arguments."""
    with pytest.raises(TypeError):
        CSVReader()


def test_invalid_measurement_path(valid_controls_file: str) -> None:
    """Test that invalid measurement file path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError) as e:
        CSVReader("invalid/path/measurements.csv", valid_controls_file)
    assert "Measurements file not found:" in str(e.value)


def test_invalid_controls_path(valid_measurements_file: str) -> None:
    """Test that invalid controls file path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError) as e:
        reader = CSVReader(valid_measurements_file, "invalid/path/controls.csv")
    assert "Controls file not found:" in str(e.value)


def test_raise_validation_mode(
    valid_measurements_file: str, valid_controls_file: str
) -> None:
    """Test that default validation mode is 'raise'."""
    reader = CSVReader(valid_measurements_file, valid_controls_file)
    assert reader._validation_mode == "raise"


def test_skip_validation_mode(
    valid_measurements_file: str, valid_controls_file: str
) -> None:
    """Test that validation_mode='skip' sets the correct mode."""
    reader = CSVReader(
        valid_measurements_file, valid_controls_file, validation_mode="skip"
    )
    assert reader._validation_mode == "skip"


def test_none_validation_mode(
    valid_measurements_file: str, valid_controls_file: str
) -> None:
    """Test that validation_mode=None sets the correct mode."""
    reader = CSVReader(
        valid_measurements_file, valid_controls_file, validation_mode=None
    )
    assert reader._validation_mode is None


def test_invalid_validation_mode(
    valid_measurements_file: str, valid_controls_file: str
) -> None:
    """Test that invalid validation_mode raises ValueError."""
    with pytest.raises(ValueError):
        CSVReader(
            valid_measurements_file,
            valid_controls_file,
            validation_mode="invalid_mode",
        )


def test_measurements_columns_name_standarization(
    measurements_file_columns_to_rename: str, valid_controls_file: str
) -> None:
    """Test that measurement columns are standardized to expected names."""
    reader = CSVReader(measurements_file_columns_to_rename, valid_controls_file)
    reader.read_measurements()

    old_stn_names = ["STN_PK", "STN_ID", "STN_H", "STN_SH"]
    new_stn_names = ["stn_pk", "stn_id", "stn_h", "stn_sh"]

    old_meas_names = [
        "TRG_ID",
        "TRG_H",
        "TRG_SH",
        "SD",
        "SSD",
        "HD",
        "SHD",
        "VD",
        "SVD",
        "DX",
        "SDX",
        "DY",
        "SDY",
        "DZ",
        "SDZ",
        "A",
        "SA",
        "HZ",
        "SHZ",
    ]
    new_meas_names = [
        "trg_id",
        "trg_h",
        "trg_sh",
        "sd",
        "ssd",
        "hd",
        "shd",
        "vd",
        "svd",
        "dx",
        "sdx",
        "dy",
        "sdy",
        "dz",
        "sdz",
        "a",
        "sa",
        "hz",
        "shz",
    ]

    for old_name, new_name in zip(old_stn_names, new_stn_names):
        assert old_name not in reader.stations.columns
        assert new_name in reader.stations.columns

    for old_name, new_name in zip(old_meas_names, new_meas_names):
        assert old_name not in reader.measurements.columns
        assert new_name in reader.measurements.columns


def test_controls_columns_name_standarization_e_n_el(
    valid_measurements_file: str, controls_file_column_to_rename_e_n_el: str
) -> None:
    """Test that controls columns E, N, EL are renamed to x, y, z."""
    reader = CSVReader(valid_measurements_file, controls_file_column_to_rename_e_n_el)
    reader.read_controls()

    old_names = ["NR", "E", "N", "EL", "SE", "SN", "SEL"]
    new_names = ["id", "x", "y", "z", "sx", "sy", "sz"]

    for old_name, new_name in zip(old_names, new_names):
        assert old_name not in reader.controls.columns
        assert new_name in reader.controls.columns


def test_controls_columns_name_standarization_easting_northing_height(
    valid_measurements_file: str,
    controls_file_column_to_rename_easting_northing_height: str,
) -> None:
    """Test that controls columns EASTING, NORTHING, HEIGHT are renamed to x, y, z."""
    reader = CSVReader(
        valid_measurements_file,
        controls_file_column_to_rename_easting_northing_height,
    )
    reader.read_controls()

    old_names = ["NR", "EASTING", "NORTHING", "HEIGHT", "SE", "SN", "SH"]
    new_names = ["id", "x", "y", "z", "sx", "sy", "sz"]

    for old_name, new_name in zip(old_names, new_names):
        assert old_name not in reader.controls.columns
        assert new_name in reader.controls.columns


def test_measurements_file_filtering(
    measurements_file_to_filter: str, valid_controls_file: str
) -> None:
    """Test that unnecessary columns are filtered from measurements file."""
    reader = CSVReader(measurements_file_to_filter, valid_controls_file)
    reader.read_measurements()

    assert "UNNECESSARY COLUMN" not in reader.measurements.columns
    assert "EXTRA COLUMN" not in reader.measurements.columns


def test_controls_file_filtering(
    valid_measurements_file: str, controls_file_to_filter: str
) -> None:
    """Test that unnecessary columns are filtered from controls file."""
    reader = CSVReader(valid_measurements_file, controls_file_to_filter)
    reader.read_controls()

    assert "UNNECESSARY COLUMN" not in reader.controls.columns
    assert "EXTRA COLUMN" not in reader.controls.columns


def test_measurements_missing_mandatory_columns(
    measurements_file_missing_mandatory_columns: str, valid_controls_file: str
) -> None:
    """Test that missing mandatory columns in measurements raises error."""
    reader = CSVReader(measurements_file_missing_mandatory_columns, valid_controls_file)
    with pytest.raises(MissingMandatoryColumnsError):
        reader.read_measurements()


def test_controls_missing_mandatory_columns(
    valid_measurements_file: str, controls_file_missing_mandatory_columns: str
) -> None:
    """Test that missing mandatory columns in controls raises error."""
    reader = CSVReader(valid_measurements_file, controls_file_missing_mandatory_columns)
    with pytest.raises(MissingMandatoryColumnsError):
        reader.read_controls()


@pytest.fixture
def measurement_assertions() -> List[Tuple[int, str, str]]:
    """Fixture for measurement assertions."""
    return [
        (0, "hz", "Invalid type"),
        (0, "sdx", "Invalid type"),
        (0, "sdy", "-0.01"),
        (0, "svz", "-0.01"),
        (1, "a", "Invalid type"),
        (1, "dy", "Invalid type"),
        (1, "hd", "-100.0"),
        (1, "sdx", "-0.008"),
        (1, "ssd", "-0.01"),
        (1, "svd", "Invalid type"),
        (1, "svh", "Invalid type"),
        (1, "trg_h", "Invalid type"),
        (1, "vz", "Invalid type"),
        (2, "hd", "Invalid type"),
        (2, "sa", "-0.01"),
        (2, "sd", "Invalid type"),
        (2, "sdy", "Invalid type"),
        (2, "sdz", "Invalid type"),
        (2, "shd", "Invalid type"),
        (2, "shz", "-0.002"),
        (2, "svd", "-0.01"),
        (2, "svh", "-0.1"),
        (2, "trg_sh", "Invalid type"),
        (3, "dz", "Invalid type"),
        (3, "sd", "-100.0"),
        (3, "shd", "-0.01"),
        (3, "svz", "Invalid type"),
        (3, "vd", "Invalid type"),
        (4, "ssd", "Invalid type"),
        (4, "trg_sh", "-0.01"),
        (4, "vh", "Invalid type"),
    ]


def test_measurements_data_validation_none(
    invalid_measurements_file: str,
    valid_controls_file: str,
    measurement_assertions: List[Tuple[int, str, str]],
) -> None:
    """Test measurements data validation with validation_mode=None."""
    reader = CSVReader(
        invalid_measurements_file, valid_controls_file, validation_mode=None
    )
    with pytest.raises(ValueError):
        reader.read_measurements()

    column_names = [
        "trg_h",
        "trg_sh",
        "sd",
        "ssd",
        "hd",
        "shd",
        "vd",
        "svd",
        "dx",
        "sdx",
        "dy",
        "sdy",
        "dz",
        "sdz",
        "a",
        "sa",
        "hz",
        "shz",
        "vz",
        "svz",
        "vh",
        "svh",
    ]

    col_idx = {name: reader.measurements.columns.get_loc(name) for name in column_names}

    for row, col_name, expected in measurement_assertions:
        col = col_idx[col_name]
        assert reader.measurements.iat[row, col] == expected


def test_measurements_data_validation_skip(
    invalid_measurements_file: str,
    valid_controls_file: str,
    measurement_assertions: List[Tuple[int, str, str]],
) -> None:
    """Test measurements data validation with validation_mode='skip'."""
    reader = CSVReader(
        invalid_measurements_file, valid_controls_file, validation_mode="skip"
    )
    with pytest.warns():
        reader.read_measurements()

    column_names = [
        "trg_h",
        "trg_sh",
        "sd",
        "ssd",
        "hd",
        "shd",
        "vd",
        "svd",
        "dx",
        "sdx",
        "dy",
        "sdy",
        "dz",
        "sdz",
        "a",
        "sa",
        "hz",
        "shz",
        "vz",
        "svz",
        "vh",
        "svh",
    ]

    col_idx = {name: reader.measurements.columns.get_loc(name) for name in column_names}

    for row, col_name, _ in measurement_assertions:
        col = col_idx[col_name]
        assert pd.isna(reader.measurements.iat[row, col])


def test_measurements_data_validation_raise(
    invalid_measurements_file: str, valid_controls_file: str
) -> None:
    """Test measurements data validation with validation_mode='raise'."""
    reader = CSVReader(invalid_measurements_file, valid_controls_file)
    with pytest.raises(InvalidDataError):
        reader.read_measurements()


@pytest.fixture
def ctrl_assertions() -> List[Tuple[int, str, str]]:
    """Fixture for controls assertions."""
    return [
        (0, "sx", "Invalid type"),
        (0, "sz", "Invalid type"),
        (0, "x", "Invalid type"),
        (1, "y", "Invalid type"),
        (2, "sx", "-0.01"),
        (2, "sy", "Invalid type"),
        (2, "sz", "-0.01"),
        (2, "z", "Invalid type"),
        (3, "sy", "-0.015"),
    ]


def test_controls_data_validation_none(
    valid_measurements_file: str,
    invalid_controls_file: str,
    ctrl_assertions: List[Tuple[int, str, str]],
) -> None:
    """Test controls data validation with validation_mode=None."""
    reader = CSVReader(
        valid_measurements_file, invalid_controls_file, validation_mode=None
    )
    with pytest.raises(ValueError):
        reader.read_controls()

    col_names = ["x", "y", "z", "sx", "sy", "sz"]
    col_idx = {name: reader.controls.columns.get_loc(name) for name in col_names}

    for row, col_name, expected in ctrl_assertions:
        col = col_idx[col_name]
        assert reader.controls.iat[row, col] == expected


def test_controls_data_validation_skip(
    valid_measurements_file: str,
    invalid_controls_file: str,
    ctrl_assertions: List[Tuple[int, str, str]],
) -> None:
    """Test controls data validation with validation_mode='skip'."""
    reader = CSVReader(
        valid_measurements_file, invalid_controls_file, validation_mode="skip"
    )
    with pytest.warns():
        reader.read_controls()

    col_names = ["x", "y", "z", "sx", "sy", "sz"]
    col_idx = {name: reader.controls.columns.get_loc(name) for name in col_names}

    for row, col_name, _ in ctrl_assertions:
        col = col_idx[col_name]
        assert pd.isna(reader.controls.iat[row, col])


def test_controls_data_validation_raise(
    valid_measurements_file: str, invalid_controls_file: str
) -> None:
    """Test controls data validation with validation_mode='raise'."""
    reader = CSVReader(valid_measurements_file, invalid_controls_file)
    with pytest.raises(InvalidDataError):
        reader.read_controls()


def test_measurements_to_float(
    valid_measurements_file: str, valid_controls_file: str
) -> None:
    """Test that all measurement columns except stn_pk and trg_id are float."""
    reader = CSVReader(valid_measurements_file, valid_controls_file)
    reader.read_measurements()

    for col in reader.measurements.columns:
        if col in ["stn_pk", "trg_id"]:
            continue
        assert reader.measurements[col].dtype == float


def test_controls_to_float(
    valid_measurements_file: str, valid_controls_file: str
) -> None:
    """Test that all controls columns except id are float."""
    reader = CSVReader(valid_measurements_file, valid_controls_file)
    reader.read_controls()

    for col in reader.controls.columns:
        if col == "id":
            continue
        assert reader.controls[col].dtype == float


def test_import_empty_measurements_file_raise(
    empty_file: str, valid_controls_file: str
) -> None:
    """Test that importing an empty measurements file raises error."""
    reader = CSVReader(empty_file, valid_controls_file)
    with pytest.raises(EmptyDatasetError):
        reader.read_measurements()


def test_import_empty_measurements_file_skip(
    empty_file: str, valid_controls_file: str
) -> None:
    """Test that importing an empty measurements file with skip raises error."""
    reader = CSVReader(empty_file, valid_controls_file, validation_mode="skip")
    with pytest.raises(EmptyDatasetError):
        reader.read_measurements()


def test_import_empty_measurements_file_none(
    empty_file: str, valid_controls_file: str
) -> None:
    """Test that importing an empty measurements file with None returns empty DataFrame."""
    reader = CSVReader(empty_file, valid_controls_file, validation_mode=None)
    reader.read_measurements()
    assert reader.measurements.empty


def test_import_empty_controls_file_raise(
    valid_measurements_file: str, empty_file: str
) -> None:
    """Test that importing an empty controls file raises error."""
    reader = CSVReader(valid_measurements_file, empty_file)
    with pytest.raises(EmptyDatasetError):
        reader.read_controls()


def test_import_empty_controls_file_skip(
    valid_measurements_file: str, empty_file: str
) -> None:
    """Test that importing an empty controls file with skip raises error."""
    reader = CSVReader(valid_measurements_file, empty_file, validation_mode="skip")
    with pytest.raises(EmptyDatasetError):
        reader.read_controls()


def test_import_empty_controls_none(
    valid_measurements_file: str, empty_file: str
) -> None:
    """Test that importing an empty controls file with None returns empty DataFrame."""
    reader = CSVReader(valid_measurements_file, empty_file, validation_mode=None)
    reader.read_controls()
    assert reader.controls.empty


def test_import_measurements_first(
    valid_measurements_file: str, valid_controls_file: str
) -> None:
    """Test that reading measurements creates stations dataset as well."""
    reader = CSVReader(valid_measurements_file, valid_controls_file)
    reader.read_measurements()
    assert reader.measurements is not None
    assert reader.stations is not None


def test_import_stations_first(
    valid_measurements_file: str, valid_controls_file: str
) -> None:
    """Test that reading stations creates measurements dataset as well."""
    reader = CSVReader(valid_measurements_file, valid_controls_file)
    reader.read_stations()
    assert reader.measurements is not None
    assert reader.stations is not None


def test_get_dataset(valid_measurements_file: str, valid_controls_file: str) -> None:
    """Test that get dataset returns proper dataset."""
    reader = CSVReader(valid_measurements_file, valid_controls_file)
    reader.read_measurements()
    reader.read_controls()

    with pytest.raises(KeyError):
        reader.get_dataset("Invalid_value")

    assert "trg_id" in reader.get_dataset("Measurements")
    assert "stn_id" in reader.get_dataset("Stations")
    assert "id" in reader.get_dataset("Controls")
