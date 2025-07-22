# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

from typing import Dict

import pandas as pd
import pytest

from pysurv.data import Measurements


@pytest.fixture
def test_data() -> Dict[str, list]:
    """Return test data for measurements."""
    return {
        "stn_pk": [0, 0, 1],
        "trg_id": ["T2", "T3", "T1"],
        "hz": [0.0000, 100.0000, 200.0000],
        "vz": [0.0000, 100.0000, 200.0000],
    }


def test_set_index(test_data: dict) -> None:
    """Test that index columns are set correctly during initialization."""
    measurements = Measurements(test_data)
    assert "stn_pk" in measurements.index.names
    assert "trg_id" in measurements.index.names


def test_angle_conversion(test_data: dict, angle_units: list, rho: dict) -> None:
    """Test angle conversion during initialization."""
    for unit in angle_units:
        measurements = Measurements(test_data, angle_unit=unit)
        for col in ["hz", "vz"]:
            assert measurements.at[(0, "T2"), col] == 0.0000
            assert measurements.at[(0, "T3"), col] == 100.0000 / rho[unit]
            assert measurements.at[(1, "T1"), col] == 200.0000 / rho[unit]


def test_copy(test_data: dict) -> None:
    """Test copying Measurements returns a new instance returns proper columns."""
    measurements = Measurements(test_data)
    measurements_copy = measurements.copy()

    assert isinstance(measurements_copy, Measurements)
    assert measurements is not measurements_copy


def test_linear_measurement_columns(valid_measurements_data: dict) -> None:
    """Test linear measurement columns property returns proper columns."""
    measurements = Measurements(valid_measurements_data)

    linear_measurement_columns = {"sd", "hd", "vd", "dx", "dy", "dz"}

    assert not measurements.linear_measurement_columns.has_duplicates
    assert set(measurements.linear_measurement_columns) == linear_measurement_columns


def test_linear_sigma_columns(valid_measurements_data: dict) -> None:
    """Test linear sigma columns property returns proper columns."""
    measurements = Measurements(valid_measurements_data)

    linear_sigma_columns = {"ssd", "shd", "svd", "sdx", "sdy", "sdz"}

    assert not measurements.linear_sigma_columns.has_duplicates
    assert set(measurements.linear_sigma_columns) == linear_sigma_columns


def test_linear_columns(valid_measurements_data: dict) -> None:
    """Test linear columns property returns proper columns."""
    measurements = Measurements(valid_measurements_data)

    linear_measurement_columns = {"sd", "hd", "vd", "dx", "dy", "dz"}
    linear_sigma_columns = {"ssd", "shd", "svd", "sdx", "sdy", "sdz"}
    linear_columns = linear_measurement_columns.union(linear_sigma_columns)

    assert not measurements.linear_columns.has_duplicates
    assert set(measurements.linear_columns) == linear_columns


def test_angular_measurement_columns(valid_measurements_data: dict) -> None:
    """Test angular measurement columns property returns proper columns."""
    measurements = Measurements(valid_measurements_data)

    angular_measurement_columns = {"a", "hz", "vz", "vh"}

    assert not measurements.angular_measurement_columns.has_duplicates
    assert set(measurements.angular_measurement_columns) == angular_measurement_columns


def test_angular_sigma_columns(valid_measurements_data: dict) -> None:
    """Test angular sigma columns property returns proper columns."""
    measurements = Measurements(valid_measurements_data)

    angular_sigma_columns = {"sa", "shz", "svz", "svh"}

    assert not measurements.angular_sigma_columns.has_duplicates
    assert set(measurements.angular_sigma_columns) == angular_sigma_columns


def test_angular_columns(valid_measurements_data: dict) -> None:
    """Test angular columns property returns proper columns."""
    measurements = Measurements(valid_measurements_data)

    angular_measurement_columns = {"a", "hz", "vz", "vh"}
    angular_sigma_columns = {"sa", "shz", "svz", "svh"}
    angular_columns = angular_measurement_columns.union(angular_sigma_columns)

    assert not measurements.angular_columns.has_duplicates
    assert set(measurements.angular_columns) == angular_columns


def test_measurement_columns(valid_measurements_data: dict) -> None:
    """Test measurement columns property returns proper columns."""
    measurements = Measurements(valid_measurements_data)

    linear_measurement_columns = {"sd", "hd", "vd", "dx", "dy", "dz"}
    angular_measurement_columns = {"a", "hz", "vz", "vh"}
    measurement_columns = linear_measurement_columns.union(angular_measurement_columns)

    assert not measurements.measurement_columns.has_duplicates
    assert set(measurements.measurement_columns) == measurement_columns


def test_sigma_columns(valid_measurements_data: dict) -> None:
    """Test sigma columns property returns proper columns."""
    measurements = Measurements(valid_measurements_data)

    linear_sigma_columns = {"ssd", "shd", "svd", "sdx", "sdy", "sdz"}
    angular_sigma_columns = {"sa", "shz", "svz", "svh"}
    sigma_columns = linear_sigma_columns.union(angular_sigma_columns)

    assert not measurements.sigma_columns.has_duplicates
    assert set(measurements.sigma_columns) == sigma_columns


def test_measurement_data(valid_measurements_data: dict) -> None:
    """Test measurement_data property returns correct columns and type."""
    measurements = Measurements(valid_measurements_data)

    assert set(measurements.measurement_data.columns) == set(
        measurements.measurement_columns
    )
    assert isinstance(measurements.measurement_data, Measurements)


def test_sigma_data(valid_measurements_data: dict) -> None:
    """Test sigma_data property returns correct columns and type."""
    measurements = Measurements(valid_measurements_data)

    assert set(measurements.sigma_data.columns) == set(measurements.sigma_columns)
    assert isinstance(measurements.sigma_data, Measurements)


def test_constructor_sliced(valid_measurements_data: dict) -> None:
    """Test slicing returns pandas Series."""
    measurements = Measurements(valid_measurements_data)

    assert isinstance(measurements["sd"], pd.Series)
    assert isinstance(measurements.sd, pd.Series)


def test_display(test_data: dict, angle_units: list) -> None:
    """Test display method for angle conversion."""
    measurements = Measurements(test_data)
    for unit in angle_units:
        displayed = measurements.display(angle_unit=unit)
        for col in ["hz", "vz"]:
            assert displayed.at[(0, "T2"), col] == 0.0000
            assert displayed.at[(0, "T3"), col] == 100.0000
            assert displayed.at[(1, "T1"), col] == 200.0000
