# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

from pysurv import Dataset
from pysurv.data import Controls, Measurements, Stations


def test_dataset_instance(
    valid_measurements_file: str, valid_controls_file: str
) -> None:
    """Test Dataset instance creation."""
    dataset = Dataset.from_csv(valid_measurements_file, valid_controls_file)
    assert isinstance(dataset, Dataset)


def test_dataset_measurements_instance(
    valid_measurements_file: str, valid_controls_file: str
) -> None:
    """Test dataset measurements is Measurements."""
    dataset = Dataset.from_csv(valid_measurements_file, valid_controls_file)
    assert isinstance(dataset.measurements, Measurements)


def test_dataset_controls_instance(
    valid_measurements_file: str, valid_controls_file: str
) -> None:
    """Test dataset controls is Controls."""
    dataset = Dataset.from_csv(valid_measurements_file, valid_controls_file)
    assert isinstance(dataset.controls, Controls)


def test_dataset_stations_instance(
    valid_measurements_file: str, valid_controls_file: str
) -> None:
    """Test dataset stations is Stations."""
    dataset = Dataset.from_csv(valid_measurements_file, valid_controls_file)
    assert isinstance(dataset.stations, Stations)


def test_dataset_measurements_view_columns(
    valid_measurements_file: str, valid_controls_file: str
) -> None:
    """Test dataset measurements_view columns and index."""
    dataset = Dataset.from_csv(valid_measurements_file, valid_controls_file)
    view_index = ["stn_pk", "stn_id", "stn_h", "stn_sh", "trg_id", "trg_h", "trg_sh"]
    view_columns = [
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

    for idx in view_index:
        assert idx in dataset.measurements_view.index.names

    for col in view_columns:
        assert col in dataset.measurements_view.columns
