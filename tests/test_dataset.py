import pytest

from pysurv.data import Dataset, Measurements, Controls, Stations


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
    
def test_dataset_measurements_view_columns(valid_measurements_file, valid_controls_file):
    dataset = Dataset.from_csv(valid_measurements_file, valid_controls_file)
    assert 'stn_pk' in dataset.measurements_view.index.names
    assert 'stn_id' in dataset.measurements_view.index.names
    assert 'stn_h' in dataset.measurements_view.index.names
    assert 'stn_sh' in dataset.measurements_view.index.names
    assert 'trg_id' in dataset.measurements_view.index.names
    assert 'trg_h' in dataset.measurements_view.index.names
    assert 'trg_sh' in dataset.measurements_view.index.names
    assert "sd" in dataset.measurements_view.columns
    assert "ssd" in dataset.measurements_view.columns
    assert "hd" in dataset.measurements_view.columns
    assert "shd" in dataset.measurements_view.columns
    assert "vd" in dataset.measurements_view.columns
    assert "svd" in dataset.measurements_view.columns
    assert "dx" in dataset.measurements_view.columns
    assert "sdx" in dataset.measurements_view.columns
    assert "dy" in dataset.measurements_view.columns
    assert "sdy" in dataset.measurements_view.columns
    assert "dz" in dataset.measurements_view.columns
    assert "sdz" in dataset.measurements_view.columns
    assert "a" in dataset.measurements_view.columns
    assert "sa" in dataset.measurements_view.columns
    assert "hz" in dataset.measurements_view.columns
    assert "shz" in dataset.measurements_view.columns
    assert "vz" in dataset.measurements_view.columns
    assert "svz" in dataset.measurements_view.columns
    assert "vh" in dataset.measurements_view.columns
    assert "svh" in dataset.measurements_view.columns