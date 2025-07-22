# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

import numpy as np
import pandas as pd
import pytest

from pysurv.data import Stations


@pytest.fixture
def test_data() -> dict[str, list]:
    """Returns test data for creating Stations dataset."""
    return {
        "stn_pk": [0, 1, 2],
        "stn_id": ["stn_1", "stn_2", "stn_3"],
        "stn_h": [1.653, 1.234, 0.0],
        "stn_sh": [0.01, 0.01, 0.002],
    }


def test_set_index(test_data: dict[str, list]) -> None:
    """Test that the index is set to 'stn_pk' on init."""
    stations = Stations(test_data)

    assert stations.index.name == "stn_pk"


def test_copy(test_data: dict[str, list]) -> None:
    """Test that Stations.copy() returns a new Stations instance."""
    stations = Stations(test_data)
    stations_copy = stations.copy()

    assert isinstance(stations_copy, Stations)
    assert stations is not stations_copy


def test_append_orientation_contant(test_data: dict[str, list]) -> None:
    """Test appending orientation constant to stations."""
    hz_data = pd.DataFrame(
        {
            "stn_pk": [0, 1],
            "trg_id": ["stn_2", "stn_3"],
            "hz": [0.0000, np.pi / 2],
        }
    ).set_index(["stn_pk", "trg_id"])

    ctrl_data = pd.DataFrame(
        {
            "id": ["stn_1", "stn_2", "stn_3"],
            "x": [0, 100, 100],
            "y": [0, 0, 100],
        }
    ).set_index("id")

    stations = Stations(test_data)
    stations.append_orientation_constant(hz_data, ctrl_data)

    assert stations.at[0, "orientation"] == 0.0000
    assert stations.at[1, "orientation"] == 0.0000
    assert pd.isna(stations.at[2, "orientation"])


def test_display_with_no_ang(test_data: dict[str, list]) -> None:
    """Test display method with no 'orientation' column."""
    stations = Stations(test_data)
    disp = stations.display()

    assert disp.index.name == "stn_pk"


def test_display_with_ang(
    test_data: dict[str, list], angle_units: tuple[str], rho: dict[str:float]
) -> None:
    """Test display method with 'orientation' column."""

    test_data.update({"orientation": [0.0000, np.pi, pd.NA]})
    stations_orientation = Stations(test_data)

    for unit in angle_units:
        disp_orientation = stations_orientation.display(angle_unit=unit)

        assert disp_orientation.at[0, "orientation"] == 0.0000
        assert disp_orientation.at[1, "orientation"] == np.pi * rho[unit]
        assert pd.isna(disp_orientation.at[2, "orientation"])
