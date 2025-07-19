import pytest

from pysurv import Dataset
from pysurv.data import Controls, Measurements, Stations


def test_dataset_instance(valid_measurements_file, valid_controls_file):
    dataset = Dataset.from_csv(valid_measurements_file, valid_controls_file)
    assert isinstance(dataset, Dataset)


def test_dataset_measurements_instance(valid_measurements_file, valid_controls_file):
    dataset = Dataset.from_csv(valid_measurements_file, valid_controls_file)
    assert isinstance(dataset.measurements, Measurements)


def test_dataset_controls_instance(valid_measurements_file, valid_controls_file):
    dataset = Dataset.from_csv(valid_measurements_file, valid_controls_file)
    assert isinstance(dataset.controls, Controls)


def test_dataset_stations_instance(valid_measurements_file, valid_controls_file):
    dataset = Dataset.from_csv(valid_measurements_file, valid_controls_file)
    assert isinstance(dataset.stations, Stations)


def test_dataset_measurements_view_columns(
    valid_measurements_file, valid_controls_file
):
    dataset = Dataset.from_csv(valid_measurements_file, valid_controls_file)
    view_index = ["stn_pk", "stn_id", "stn_h", "stn_sh", "trg_id", "trg_h", "trg_sh"]
    view_columns = ["sd", "ssd", "hd", "shd", "vd", "svd", 
                    "dx", "sdx","dy", "sdy", "dz", "sdz", 
                    "a", "sa", "hz", "shz", "vz", "svz", "vh", "svh"]
    
    for idx in view_index:
        assert idx in dataset.measurements_view.index.names
        
    for col in view_columns:
        assert col in dataset.measurements_view.columns