import os
import tempfile

import pandas as pd
import pytest


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def empty_data():
    data = {"id": [], "stn_id": [], "trg_id": []}
    yield pd.DataFrame(data)


@pytest.fixture
def empty_file(empty_data, temp_dir):
    file_path = os.path.join(temp_dir, "empty.csv")
    empty_data.to_csv(file_path, index=False)
    yield file_path


@pytest.fixture
def valid_measurements_data():
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
    yield pd.DataFrame(data)


@pytest.fixture
def valid_measurements_file(valid_measurements_data, temp_dir):
    file_path = os.path.join(temp_dir, "valid_measurements.csv")
    valid_measurements_data.to_csv(file_path, index=False)
    yield file_path


@pytest.fixture
def valid_controls_data():
    data = {
        "id": ["C1", "C2", "C3", "C4", "C5"],
        "x": [1000.00, 2000.00, 3000.00, 4000.00, 5000.00],
        "y": [1000.00, 2000.00, 3000.00, 4000.00, 5000.00],
        "z": [100.00, 100.10, 100.20, None, 100.40],
        "sx": [-1, 0.000, 0.010, None, -1],
        "sy": [-1, 0.000, -1, 0.015, 0.000],
        "sz": [-1, -1, 0.010, None, 0.000],
    }
    yield pd.DataFrame(data)


@pytest.fixture
def valid_controls_file(valid_controls_data, temp_dir):
    file_path = os.path.join(temp_dir, "valid_controls.csv")
    valid_controls_data.to_csv(file_path, index=False)
    yield file_path


@pytest.fixture
def invalid_measurements_data():
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
        "shz": [None, None, -0.0020, "Invalid type", None],
        "vz": [0.0000, "Invalid type", 200.0000, 300.0000, None],
        "svz": [-0.0100, 0.0030, None, "Invalid type", None],
        "vh": [-100.0000, 0.0000, 100.0000, None, "Invalid type"],
        "svh": [0.0500, "Invalid type", -0.1000, None, None],
    }
    yield pd.DataFrame(data)


@pytest.fixture
def invalid_measurements_file(invalid_measurements_data, temp_dir):
    file_path = os.path.join(temp_dir, "invalid_measurements.csv")
    invalid_measurements_data.to_csv(file_path, index=False)
    yield file_path


@pytest.fixture
def invalid_controls_data():
    data = {
        "id": ["C1", "C2", "C3", "C4", "C5"],
        "x": ["Invalid type", 2000.00, 3000.00, 4000.00, 5000.00],
        "y": [1000.00, "Invalid type", 3000.00, 4000.00, 5000.00],
        "z": [100.00, 100.10, "Invalid type", None, 100.40],
        "sx": ["Invalid type", 0.000, -0.010, None, -1],
        "sy": [-1, 0.010, "Invalid type", -0.015, 0.000],
        "sz": ["Invalid type", -1, -0.010, None, 0.000],
    }
    yield pd.DataFrame(data)


@pytest.fixture
def invalid_controls_file(invalid_controls_data, temp_dir):
    file_path = os.path.join(temp_dir, "invalid_controls.csv")
    invalid_controls_data.to_csv(file_path, index=False)
    yield file_path


@pytest.fixture
def measurements_data_columns_to_rename():
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
    yield pd.DataFrame(data)


@pytest.fixture
def measurements_file_columns_to_rename(measurements_data_columns_to_rename, temp_dir):
    file_path = os.path.join(temp_dir, "measurements_to_rename.csv")
    measurements_data_columns_to_rename.to_csv(file_path, index=False)
    yield file_path


@pytest.fixture
def controls_data_column_to_rename_e_n_el():
    data = {
        "NR": ["C1", "C2", "C3", "C4", "C5"],
        "E": [1000.00, 2000.00, 3000.00, 4000.00, 5000.00],
        "N": [1000.00, 2000.00, 3000.00, 4000.00, 5000.00],
        "EL": [100.00, 100.10, 100.20, None, 100.40],
        "SE": [-1, 0.000, 0.010, None, -1],
        "SN": [-1, 0.000, -1, 0.015, 0.000],
        "SEL": [-1, -1, 0.010, None, 0.000],
    }
    yield pd.DataFrame(data)


@pytest.fixture
def controls_file_column_to_rename_e_n_el(
    controls_data_column_to_rename_e_n_el, temp_dir
):
    file_path = os.path.join(temp_dir, "controls_e_n_el.csv")
    controls_data_column_to_rename_e_n_el.to_csv(file_path, index=False)
    yield file_path


@pytest.fixture
def controls_data_column_to_rename_easting_northing_height():
    data = {
        "NR": ["C1", "C2", "C3", "C4", "C5"],
        "EASTING": [1000.00, 2000.00, 3000.00, 4000.00, 5000.00],
        "NORTHING": [1000.00, 2000.00, 3000.00, 4000.00, 5000.00],
        "HEIGHT": [100.00, 100.10, 100.20, None, 100.40],
        "SE": [-1, 0.000, 0.010, None, -1],
        "SN": [-1, 0.000, -1, 0.015, 0.000],
        "SH": [-1, -1, 0.010, None, 0.000],
    }
    yield pd.DataFrame(data)


@pytest.fixture
def controls_file_column_to_rename_easting_northing_height(
    controls_data_column_to_rename_easting_northing_height, temp_dir
):
    file_path = os.path.join(temp_dir, "controls_e_n_el.csv")
    controls_data_column_to_rename_easting_northing_height.to_csv(
        file_path, index=False
    )
    yield file_path


@pytest.fixture
def measurements_data_to_filter():
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
    yield pd.DataFrame(data)


@pytest.fixture
def measurements_file_to_filter(measurements_data_to_filter, temp_dir):
    file_path = os.path.join(temp_dir, "measurements_to_filter.csv")
    measurements_data_to_filter.to_csv(file_path, index=False)
    yield file_path


@pytest.fixture
def controls_data_to_filter():
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
    yield pd.DataFrame(data)


@pytest.fixture
def controls_file_to_filter(controls_data_to_filter, temp_dir):
    file_path = os.path.join(temp_dir, "controls_to_filter.csv")
    controls_data_to_filter.to_csv(file_path, index=False)
    yield file_path


@pytest.fixture
def measurements_data_missing_mandatory_columns():
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
    yield pd.DataFrame(data)


@pytest.fixture
def measurements_file_missing_mandatory_columns(
    measurements_data_missing_mandatory_columns, temp_dir
):
    file_path = os.path.join(temp_dir, "measurements_missing_columns.csv")
    measurements_data_missing_mandatory_columns.to_csv(file_path, index=False)
    yield file_path


@pytest.fixture
def controls_data_missing_mandatory_columns():
    data = {
        "x": [1000.00, 2000.00, 3000.00, 4000.00, 5000.00],
        "y": [1000.00, 2000.00, 3000.00, 4000.00, 5000.00],
        "z": [100.00, 100.10, 100.20, None, 100.40],
        "sx": [-1, 0.000, 0.010, None, -1],
        "sy": [-1, 0.000, -1, 0.015, 0.000],
        "sz": [-1, -1, 0.010, None, 0.000],
    }
    yield pd.DataFrame(data)


@pytest.fixture
def controls_file_missing_mandatory_columns(
    controls_data_missing_mandatory_columns, temp_dir
):
    file_path = os.path.join(temp_dir, "controls_missing_columns.csv")
    controls_data_missing_mandatory_columns.to_csv(file_path, index=False)
    yield file_path
