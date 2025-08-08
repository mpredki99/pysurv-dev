# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

import os
import tempfile
from typing import Generator, List, Tuple

import numpy as np
import pandas as pd
import pytest

from pysurv import Dataset, config
from pysurv.adjustment import DenseMatrices, config_sigma, config_solver
from pysurv.adjustment.matrices import Matrices
from pysurv.adjustment.adjustment_method_manager import AdjustmentMethodManager


# Fixtures for restoring original state config objects
@pytest.fixture(autouse=True)
def reset_config_state():
    """Restore config to its default state after each test."""
    original_angle_unit = config.angle_unit
    yield
    config.angle_unit = original_angle_unit


@pytest.fixture(autouse=True)
def reset_config_sigma_state():
    """Reset sigma_config to its default state after each test."""
    default_index = config_sigma.default_index
    yield
    config_sigma.default_index = default_index
    for idx in config_sigma.index:
        if idx == "default":
            continue
        delattr(config_sigma, idx)
    config_sigma.restore_default()


@pytest.fixture(autouse=True)
def reset_config_solver_state():
    """Reset sigma_config to its default state after each test."""
    default_index = config_solver.default_index
    yield
    config_solver.default_index = default_index
    for idx in config_solver.index:
        if idx == "default":
            continue
        delattr(config_solver, idx)
    config_solver.restore_default()


# Fixtures for testing angles in different units
@pytest.fixture
def angle_units() -> tuple[str]:
    """Returns list of angle units."""
    return ("rad", "grad", "gon", "deg")


@pytest.fixture
def rho() -> dict[str, float]:
    """Returns a dictionary of angle unit conversion factors."""
    return {
        "rad": 1,
        "grad": 200 / np.pi,
        "gon": 200 / np.pi,
        "deg": 180 / np.pi,
    }


# Fixtures for testing measurements dataset
@pytest.fixture
def valid_measurement_data() -> pd.DataFrame:
    """Returns a DataFrame with valid measurement data."""
    data = {
        "stn_id": ["C1", None, None, None, "C2"],
        "stn_h": [1.500, -1.576, None, None, None],
        "stn_sh": [0.01, None, 0.05, None, None],
        "trg_id": ["C2", "C3", "C4", "C5", "C1"],
        "trg_h": [-1.245, None, 0.000, None, 1.753],
        "trg_sh": [0.000, None, None, None, 0.01],
        "sd": [100.00, 150.00, 200.00, None, None],
        "ssd": [None, 0.01, 0.00, None, None],
        "hd": [None, None, 100.00, 150.00, 200.00],
        "shd": [None, None, None, 0.01, 0.00],
        "vd": [10.000, None, 12.120, None, -8.123],
        "svd": [0.010, None, 0.000, None, None],
        "dx": [500.000, -20.000, 0.000, -35.000, None],
        "sdx": [None, 0.008, 0.000, 0.050, None],
        "dy": [0.000, 950.000, -89.000, None, 820.000],
        "sdy": [0.010, None, 0.003, None, 0.002],
        "dz": [-800.000, 0.000, 700.000, 0.000, None],
        "sdz": [0.000, 0.001, 0.250, None, None],
        "a": [0.0000, 100.0000, 200.0000, 300.0000, None],
        "sa": [0.0001, None, 0.0100, 0.1000, None],
        "hz": [123.4567, None, 345.6789, 135.7890, 258.1369],
        "shz": [None, None, 0.0020, 0.0050, None],
        "vz": [0.0000, 100.0000, 200.0000, 300.0000, None],
        "svz": [0.0100, 0.0030, None, 0.0015, None],
        "vh": [-100.0000, 0.0000, 100.0000, None, 50.0000],
        "svh": [0.0500, 0.0010, 0.1000, None, None],
    }
    return pd.DataFrame(data)


@pytest.fixture
def invalid_measurement_data() -> pd.DataFrame:
    """Returns a DataFrame with invalid measurement data."""
    data = {
        "stn_id": ["C1", None, None, None, "C2"],
        "stn_h": [1.500, -1.576, "Invalid type", None, None],
        "stn_sh": [-0.01, "Invalid type", 0.05, None, None],
        "trg_id": ["C2", "C3", "C4", "C5", "C1"],
        "trg_h": [-1.245, "Invalid type", 0.000, None, 1.753],
        "trg_sh": [0.000, None, "Invalid type", None, -0.01],
        "sd": [100.00, 150.00, "Invalid type", -100.00, None],
        "ssd": [None, -0.01, 0.00, None, "Invalid type"],
        "hd": [None, -100.00, "Invalid type", 150.00, 200.00],
        "shd": [None, None, "Invalid type", -0.01, 0.00],
        "vd": [10.000, None, 12.120, "Invalid type", -8.123],
        "svd": [0.010, "Invalid type", -0.010, None, None],
        "dx": [500.000, -20.000, 0.000, "Invalid type", None],
        "sdx": ["Invalid type", -0.008, 0.000, 0.050, None],
        "dy": [0.000, "Invalid type", -89.000, None, 820.000],
        "sdy": [-0.010, None, "Invalid type", None, 0.002],
        "dz": [-800.000, 0.000, 700.000, "Invalid type", None],
        "sdz": [0.000, -0.001, "Invalid type", None, None],
        "a": [0.0000, "Invalid type", 200.0000, 300.0000, None],
        "sa": [0.0001, None, -0.0100, "Invalid type", None],
        "hz": ["Invalid type", None, 345.6789, 135.7890, 258.1369],
        "shz": [None, 0.0010, -0.0020, "Invalid type", None],
        "vz": [0.0000, "Invalid type", 200.0000, 300.0000, None],
        "svz": [-0.0100, 0.0030, None, "Invalid type", None],
        "vh": [-100.0000, 0.0000, 100.0000, None, "Invalid type"],
        "svh": [0.0500, "Invalid type", -0.1000, None, None],
    }
    return pd.DataFrame(data)


@pytest.fixture
def invalid_measurement_data_asserions() -> List[Tuple[int, str, str]]:
    """Fixture for invalid measurement data assertions."""
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


@pytest.fixture
def measurement_angles_data() -> pd.DataFrame:
    """Return simple angles test data for measurements."""
    data = {
        "stn_pk": [0, 0, 1],
        "trg_id": ["T2", "T3", "T1"],
        "hz": [0.0000, 100.0000, 200.0000],
        "vz": [0.0000, 100.0000, 200.0000],
    }
    return pd.DataFrame(data)


@pytest.fixture
def measurement_data_columns_to_rename() -> pd.DataFrame:
    """Returns a DataFrame with measurement columns to be renamed."""
    data = {
        "STN_ID": ["C1", None, None, None, "C2"],
        "STN_H": [1.500, -1.576, None, None, None],
        "STN_SH": [0.01, None, 0.05, None, None],
        "TRG_ID": ["C2", "C3", "C4", "C5", "C1"],
        "TRG_H": [-1.245, None, 0.000, None, 1.753],
        "TRG_SH": [0.000, None, None, None, 0.01],
        "SD": [100.00, 150.00, 200.00, None, None],
        "SSD": [None, 0.01, 0.00, None, None],
        "HD": [None, None, 100.00, 150.00, 200.00],
        "SHD": [None, None, None, 0.01, 0.00],
        "VD": [10.000, None, 12.120, None, -8.123],
        "SVD": [0.010, None, 0.000, None, None],
        "DX": [500.000, -20.000, 0.000, -35.000, None],
        "SDX": [None, 0.008, 0.000, 0.050, None],
        "DY": [0.000, 950.000, -89.000, None, 820.000],
        "SDY": [0.010, None, 0.003, None, 0.002],
        "DZ": [-800.000, 0.000, 700.000, 0.000, None],
        "SDZ": [0.000, 0.001, 0.250, None, None],
        "A": [0.0000, 100.0000, 200.0000, 300.0000, None],
        "SA": [0.0001, None, 0.0100, 0.1000, None],
        "HZ": [123.4567, None, 345.6789, 135.7890, 258.1369],
        "SHZ": [None, None, 0.0020, 0.0050, None],
        "VZ": [0.0000, 100.0000, 200.0000, 300.0000, None],
        "SVZ": [0.0100, 0.0030, None, 0.0015, None],
        "VH": [-100.0000, 0.0000, 100.0000, None, 50.0000],
        "SVH": [0.0500, 0.0010, 0.1000, None, None],
    }
    return pd.DataFrame(data)


@pytest.fixture
def measurement_data_to_filter() -> pd.DataFrame:
    """Returns a measurement DataFrame with extra columns."""
    data = {
        "stn_id": ["C1", None, None, None, "C2"],
        "stn_h": [1.500, -1.576, None, None, None],
        "stn_sh": [0.01, None, 0.05, None, None],
        "trg_id": ["C2", "C3", "C4", "C5", "C1"],
        "trg_h": [-1.245, None, 0.000, None, 1.753],
        "trg_sh": [0.000, None, None, None, 0.01],
        "sd": [100.00, 150.00, 200.00, None, None],
        "ssd": [None, 0.01, 0.00, None, None],
        "hd": [None, None, 100.00, 150.00, 200.00],
        "shd": [None, None, None, 0.01, 0.00],
        "vd": [10.000, None, 12.120, None, -8.123],
        "svd": [0.010, None, 0.000, None, None],
        "dx": [500.000, -20.000, 0.000, -35.000, None],
        "sdx": [None, 0.008, 0.000, 0.050, None],
        "dy": [0.000, 950.000, -89.000, None, 820.000],
        "sdy": [0.010, None, 0.003, None, 0.002],
        "dz": [-800.000, 0.000, 700.000, 0.000, None],
        "sdz": [0.000, 0.001, 0.250, None, None],
        "a": [0.0000, 100.0000, 200.0000, 300.0000, None],
        "sa": [0.0001, None, 0.0100, 0.1000, None],
        "hz": [123.4567, None, 345.6789, 135.7890, 258.1369],
        "shz": [None, None, 0.0020, 0.0050, None],
        "vz": [0.0000, 100.0000, 200.0000, 300.0000, None],
        "svz": [0.0100, 0.0030, None, 0.0015, None],
        "vh": [-100.0000, 0.0000, 100.0000, None, 50.0000],
        "svh": [0.0500, 0.0010, 0.1000, None, None],
        "UNNECESSARY COLUMN": [None, None, None, None, None],
        "EXTRA COLUMN": [1, 2, 3, 4, 5],
    }
    return pd.DataFrame(data)


@pytest.fixture
def measurement_data_missing_mandatory_columns() -> pd.DataFrame:
    """Returns DataFrame with missing mandatory measurement columns."""
    data = {
        "stn_h": [1.500, -1.576, None, None, None],
        "stn_sh": [0.01, None, 0.05, None, None],
        "trg_h": [-1.245, None, 0.000, None, 1.753],
        "trg_sh": [0.000, None, None, None, 0.01],
        "sd": [100.00, 150.00, 200.00, None, None],
        "ssd": [None, 0.01, 0.00, None, None],
        "hd": [None, None, 100.00, 150.00, 200.00],
        "shd": [None, None, None, 0.01, 0.00],
        "vd": [10.000, None, 12.120, None, -8.123],
        "svd": [0.010, None, 0.000, None, None],
        "dx": [500.000, -20.000, 0.000, -35.000, None],
        "sdx": [None, 0.008, 0.000, 0.050, None],
        "dy": [0.000, 950.000, -89.000, None, 820.000],
        "sdy": [0.010, None, 0.003, None, 0.002],
        "dz": [-800.000, 0.000, 700.000, 0.000, None],
        "sdz": [0.000, 0.001, 0.250, None, None],
        "a": [0.0000, 100.0000, 200.0000, 300.0000, None],
        "sa": [0.0001, None, 0.0100, 0.1000, None],
        "hz": [123.4567, None, 345.6789, 135.7890, 258.1369],
        "shz": [None, None, 0.0020, 0.0050, None],
        "vz": [0.0000, 100.0000, 200.0000, 300.0000, None],
        "svz": [0.0100, 0.0030, None, 0.0015, None],
        "vh": [-100.0000, 0.0000, 100.0000, None, 50.0000],
        "svh": [0.0500, 0.0010, 0.1000, None, None],
    }
    return pd.DataFrame(data)


# Fixtures for testing stations dataset
@pytest.fixture
def valid_station_data() -> pd.DataFrame:
    """Returns test data for creating Stations dataset."""
    data = {
        "stn_pk": [0, 1, 2],
        "stn_id": ["stn_1", "stn_2", "stn_3"],
        "stn_h": [1.653, 1.234, 0.0],
        "stn_sh": [0.01, 0.01, 0.002],
    }
    return pd.DataFrame(data)


# Fixtures for testing controls dataset
@pytest.fixture
def valid_control_data() -> pd.DataFrame:
    """Returns a DataFrame with valid control data."""
    data = {
        "id": ["C1", "C2", "C3", "C4", "C5"],
        "x": [1000.00, 2000.00, 3000.00, 4000.00, 5000.00],
        "y": [1000.00, 2000.00, 3000.00, 4000.00, 5000.00],
        "z": [100.00, 100.10, 100.20, None, 100.40],
        "sx": [-1, 0.000, 0.010, None, -1],
        "sy": [-1, 0.000, -1, 0.015, 0.000],
        "sz": [-1, -1, 0.010, None, 0.000],
    }
    return pd.DataFrame(data)


@pytest.fixture
def invalid_control_data() -> pd.DataFrame:
    """Returns a DataFrame with invalid control data."""
    data = {
        "id": ["C1", "C2", "C3", "C4", "C5"],
        "x": ["Invalid type", 2000.00, 3000.00, 4000.00, 5000.00],
        "y": [1000.00, "Invalid type", 3000.00, 4000.00, 5000.00],
        "z": [100.00, 100.10, "Invalid type", None, 100.40],
        "sx": ["Invalid type", 0.000, -0.010, None, -1],
        "sy": [-1, 0.010, "Invalid type", -0.015, 0.000],
        "sz": ["Invalid type", -1, -0.010, None, 0.000],
    }
    return pd.DataFrame(data)


@pytest.fixture
def invalid_control_data_assertions() -> List[Tuple[int, str, str]]:
    """Fixture for invalid control data assertions."""
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


@pytest.fixture
def control_data_without_y() -> pd.DataFrame:
    """Returns DataFrame without 'y' column for controls."""
    data = {
        "id": ["T1", "T2"],
        "x": [100.00, 100.00],
        "z": [300.00, 300.00],
        "sx": [0.01, 0.01],
        "sz": [0.02, 0.02],
    }
    return pd.DataFrame(data)


@pytest.fixture
def control_data_without_sy() -> pd.DataFrame:
    """Returns DataFrame without 'sy' column for controls."""
    data = {
        "id": ["T1", "T2"],
        "x": [100.00, 100.00],
        "y": [200.00, 200.00],
        "z": [300.00, 300.00],
        "sx": [0.01, 0.01],
        "sz": [0.02, 0.02],
    }
    return pd.DataFrame(data)


@pytest.fixture
def control_data_1D() -> pd.DataFrame:
    """Returns 1D DataFrame with only 'z' and 'sz' columns for controls."""
    data = {
        "id": ["T1", "T2", "T3"],
        "z": [100.000, 101.000, 102.000],
        "sz": [0.001, 0.001, 0.001],
    }
    return pd.DataFrame(data)


@pytest.fixture
def control_data_2D() -> pd.DataFrame:
    """Returns 2D DataFrame with 'x', 'y', 'sx', 'sy' columns for controls."""
    data = {
        "id": ["T1", "T2", "T3"],
        "x": [100.00, 200.00, 300.00],
        "y": [300.00, 200.00, 100.00],
        "sx": [0.01, 0.01, 0.01],
        "sy": [0.01, 0.01, 0.01],
    }
    return pd.DataFrame(data)


@pytest.fixture
def control_data_column_to_rename_e_n_el() -> pd.DataFrame:
    """Returns a DataFrame with E, N, EL columns for controls."""
    data = {
        "NR": ["C1", "C2", "C3", "C4", "C5"],
        "E": [1000.00, 2000.00, 3000.00, 4000.00, 5000.00],
        "N": [1000.00, 2000.00, 3000.00, 4000.00, 5000.00],
        "EL": [100.00, 100.10, 100.20, None, 100.40],
        "SE": [-1, 0.000, 0.010, None, -1],
        "SN": [-1, 0.000, -1, 0.015, 0.000],
        "SEL": [-1, -1, 0.010, None, 0.000],
    }
    return pd.DataFrame(data)


@pytest.fixture
def control_data_column_to_rename_easting_northing_height() -> pd.DataFrame:
    """Returns a DataFrame with EASTING, NORTHING, HEIGHT columns for controls."""
    data = {
        "NR": ["C1", "C2", "C3", "C4", "C5"],
        "EASTING": [1000.00, 2000.00, 3000.00, 4000.00, 5000.00],
        "NORTHING": [1000.00, 2000.00, 3000.00, 4000.00, 5000.00],
        "HEIGHT": [100.00, 100.10, 100.20, None, 100.40],
        "SE": [-1, 0.000, 0.010, None, -1],
        "SN": [-1, 0.000, -1, 0.015, 0.000],
        "SH": [-1, -1, 0.010, None, 0.000],
    }
    return pd.DataFrame(data)


@pytest.fixture
def control_data_to_filter() -> pd.DataFrame:
    """Returns a control DataFrame with extra columns."""
    data = {
        "id": ["C1", "C2", "C3", "C4", "C5"],
        "x": [1000.00, 2000.00, 3000.00, 4000.00, 5000.00],
        "y": [1000.00, 2000.00, 3000.00, 4000.00, 5000.00],
        "z": [100.00, 100.10, 100.20, None, 100.40],
        "sx": [-1, 0.000, 0.010, None, -1],
        "sy": [-1, 0.000, -1, 0.015, 0.000],
        "sz": [-1, -1, 0.010, None, 0.000],
        "UNNECESSARY COLUMN": [None, None, None, None, None],
        "EXTRA COLUMN": [1, 2, 3, 4, 5],
    }
    return pd.DataFrame(data)


@pytest.fixture
def control_data_missing_mandatory_columns() -> pd.DataFrame:
    """Returns DataFrame with missing mandatory control columns."""
    data = {
        "x": [1000.00, 2000.00, 3000.00, 4000.00, 5000.00],
        "y": [1000.00, 2000.00, 3000.00, 4000.00, 5000.00],
        "z": [100.00, 100.10, 100.20, None, 100.40],
        "sx": [-1, 0.000, 0.010, None, -1],
        "sy": [-1, 0.000, -1, 0.015, 0.000],
        "sz": [-1, -1, 0.010, None, 0.000],
    }
    return pd.DataFrame(data)


# Fixtures for testing import data
@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """Yields a temporary directory path."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def empty_data() -> pd.DataFrame:
    """Returns an empty DataFrame with id, stn_id, trg_id columns."""
    data = {"id": [], "stn_id": [], "trg_id": []}
    return pd.DataFrame(data)


@pytest.fixture
def empty_file(empty_data: pd.DataFrame, temp_dir: str) -> str:
    """Writes an empty DataFrame to a CSV file and returns its path."""
    file_path: str = os.path.join(temp_dir, "empty.csv")
    empty_data.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def valid_measurement_file(valid_measurement_data: pd.DataFrame, temp_dir: str) -> str:
    """Writes valid measurement data to a CSV file and returns its path."""
    file_path: str = os.path.join(temp_dir, "valid_measurements.csv")
    valid_measurement_data.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def valid_control_file(valid_control_data: pd.DataFrame, temp_dir: str) -> str:
    """Writes valid control data to a CSV file and returns its path."""
    file_path: str = os.path.join(temp_dir, "valid_controls.csv")
    valid_control_data.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def invalid_measurement_file(
    invalid_measurement_data: pd.DataFrame, temp_dir: str
) -> str:
    """Writes invalid measurement data to a CSV file and returns its path."""
    file_path: str = os.path.join(temp_dir, "invalid_measurements.csv")
    invalid_measurement_data.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def invalid_control_file(invalid_control_data: pd.DataFrame, temp_dir: str) -> str:
    """Writes invalid control data to a CSV file and returns its path."""
    file_path: str = os.path.join(temp_dir, "invalid_controls.csv")
    invalid_control_data.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def measurement_file_columns_to_rename(
    measurement_data_columns_to_rename: pd.DataFrame, temp_dir: str
) -> str:
    """Writes measurement data with columns to rename to a CSV file and returns its path."""
    file_path: str = os.path.join(temp_dir, "measurements_to_rename.csv")
    measurement_data_columns_to_rename.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def control_file_column_to_rename_e_n_el(
    control_data_column_to_rename_e_n_el: pd.DataFrame, temp_dir: str
) -> str:
    """Writes control data with E, N, EL columns to a CSV file and returns its path."""
    file_path: str = os.path.join(temp_dir, "controls_e_n_el.csv")
    control_data_column_to_rename_e_n_el.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def control_file_column_to_rename_easting_northing_height(
    control_data_column_to_rename_easting_northing_height: pd.DataFrame, temp_dir: str
) -> str:
    """Writes control data with EASTING, NORTHING, HEIGHT columns to a CSV file and returns its path."""
    file_path: str = os.path.join(temp_dir, "controls_e_n_el.csv")
    control_data_column_to_rename_easting_northing_height.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def measurement_file_to_filter(
    measurement_data_to_filter: pd.DataFrame, temp_dir: str
) -> str:
    """Writes measurement data with extra columns to a CSV file and returns its path."""
    file_path: str = os.path.join(temp_dir, "measurements_to_filter.csv")
    measurement_data_to_filter.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def control_file_to_filter(control_data_to_filter: pd.DataFrame, temp_dir: str) -> str:
    """Writes control data with extra columns to a CSV file and returns its path."""
    file_path: str = os.path.join(temp_dir, "controls_to_filter.csv")
    control_data_to_filter.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def measurement_file_missing_mandatory_columns(
    measurement_data_missing_mandatory_columns: pd.DataFrame, temp_dir: str
) -> str:
    """Writes measurement data with missing mandatory columns to a CSV file and returns its path."""
    file_path: str = os.path.join(temp_dir, "measurements_missing_columns.csv")
    measurement_data_missing_mandatory_columns.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def control_file_missing_mandatory_columns(
    control_data_missing_mandatory_columns: pd.DataFrame, temp_dir: str
) -> str:
    """Writes control data with missing mandatory columns to a CSV file and returns its path."""
    file_path: str = os.path.join(temp_dir, "controls_missing_columns.csv")
    control_data_missing_mandatory_columns.to_csv(file_path, index=False)
    return file_path


# Test objects
@pytest.fixture
def mock_dataset_size():
    """Fixture providing mock dataset with customizable size."""
    from unittest.mock import Mock

    def create_mock_dataset_size(size=pd.Series([0, 0, 0])):

        dataset = Mock()
        dataset.controls.memory_usage = Mock(return_value=size)
        dataset.stations.memory_usage = Mock(return_value=size)
        dataset.measurements.memory_usage = Mock(return_value=size)
        dataset.measurements.angular_measurement_columns = []

        return dataset

    return create_mock_dataset_size


@pytest.fixture
def adjustment_test_dataset():
    """Fixture providing sample dataset for performing adjustment."""
    from pysurv import Dataset

    return Dataset.from_csv("tests/measurements.csv", "tests/controls.csv")


@pytest.fixture
def MethodManagerTester():
    """Fixture providing a test MethodManagerAdjustment subclass."""
    from pysurv.adjustment.adjustment_method_manager import AdjustmentMethodManager

    class CreateMethodManagerTester(AdjustmentMethodManager):
        def __init__(
            self,
            observations: str = "weighted",
            obs_tuning_constants: dict | None = None,
            free_adjustment: str | None = None,
            free_adj_tuning_constants: dict | None = None,
        ) -> None:
            self._matrices = None

            self._observations = observations
            self._obs_tuning_constants = obs_tuning_constants
            self._free_adjustment = free_adjustment
            self._free_adj_tuning_constants = free_adj_tuning_constants

        def _get_tuning_constants(
            self, tuning_constants: dict | None, method: str | None
        ) -> None:
            pass

    return CreateMethodManagerTester


@pytest.fixture
def MatricesTester():
    """Fixture providing a test Matrices subclass."""
    from pysurv.adjustment.matrices import Matrices
    from pysurv.adjustment.adjustment_method_manager import AdjustmentMethodManager

    class CreateMatricesTester(Matrices):
        def __init__(self, methods: AdjustmentMethodManager):
            self._X = None
            self._Y = None
            self._W = None
            self._sW = None

            self._R = None
            self._sX = None

            self._k = None

            self._methods = methods
            self._methods._inject_matrices(self)

        def _build_xyw_matrices(self) -> None:
            self._X = np.zeros((30, 10))
            self._Y = np.zeros((30, 1))
            self._W = np.eye(30)

        def _build_inner_constraints_matrix(self) -> None:
            self._R = np.zeros((4, 10))

        def _build_sx_matrix(self) -> None:
            self._W = np.eye(10)

        def _build_sw_matrix(self) -> None:
            self._W = np.eye(10)

        def update_xy_matrices(self) -> None:
            pass

        def update_w_matrix(self, v: np.ndarray) -> None:
            pass

        def update_sw_matrix(self, v: np.ndarray) -> None:
            pass

    return CreateMatricesTester


@pytest.fixture
def adjustment_test_matrices(
    adjustment_test_dataset: Dataset, MethodManagerTester: AdjustmentMethodManager
) -> Matrices:
    """Return matrices object created based on test dataset."""
    methods = MethodManagerTester()
    return DenseMatrices(adjustment_test_dataset, methods)


@pytest.fixture
def strategies():
    """Returns names of the matrices construct strategies."""
    return ["speed", "memory_safe"]
