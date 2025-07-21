import numpy as np
import pandas as pd
import pytest

from pysurv.data import Stations


@pytest.fixture
def stn_data():
    return {
        "stn_pk": [0, 1, 2],
        "stn_id": ["stn_1", "stn_2", "stn_3"],
        "stn_h": [1.653, 1.234, 0.0],
        "stn_sh": [0.01, 0.01, 0.002],
    }


def test_set_index(stn_data):
    stations = Stations(stn_data)

    assert stations.index.name == "stn_pk"


def test_copy(stn_data):
    stations = Stations(stn_data)
    stations_copy = stations.copy()

    assert isinstance(stations_copy, Stations)
    assert stations is not stations_copy


def test_append_orientation_contant(stn_data):

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

    stations = Stations(stn_data)
    stations.append_orientation_constant(hz_data, ctrl_data)
    from pysurv import config

    assert stations.at[0, "orientation"] == 0.0000
    assert stations.at[1, "orientation"] == 0.0000
    assert pd.isna(stations.at[2, "orientation"])


def test_display(stn_data):
    stations = Stations(stn_data)
    disp = stations.display()

    assert disp.index.name == "stn_pk"

    stn_data.update({"orientation": [0.0000, np.pi, pd.NA]})
    stations_orientation = Stations(stn_data)

    angles = {"rad": np.pi, "grad": 200, "gon": 200, "deg": 180}
    for unit, angle in angles.items():
        disp_orientation = stations_orientation.display(angle_unit=unit)
        assert disp_orientation.at[0, "orientation"] == 0.0000
        assert disp_orientation.at[1, "orientation"] == angle
        assert pd.isna(disp_orientation.at[2, "orientation"])
