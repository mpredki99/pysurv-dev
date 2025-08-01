# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

from typing import Any

import numpy as np
import pandas as pd
import pytest

from pysurv import config
from pysurv.adjustment import config_sigma
from pysurv.adjustment._constants import DEFAULT_CONFIG_SIGMA
from pysurv.warnings import DefaultIndexWarning


@pytest.fixture
def sigma_columns() -> tuple[str]:
    """Returns list of sigma columns."""
    return (
        "stn_sh",
        "ssd",
        "shd",
        "svd",
        "sdx",
        "sdy",
        "sdz",
        "sa",
        "shz",
        "svz",
        "svh",
        "sx",
        "sy",
        "sz",
    )


def test_singleton() -> None:
    """Test that sigma_config is a singleton."""
    config_sigma_1 = config_sigma
    config_sigma_2 = config.config_sigma

    assert config_sigma_1 is config_sigma_2


def test_sigma_config_string(capsys: pytest.CaptureFixture) -> None:
    """Test string representation of sigma_config contains 'SIGMA CONFIG'."""
    print(config_sigma)
    captured = capsys.readouterr()
    assert "CONFIG SIGMA" in captured.out


def test_sigma_config_row_string(capsys: pytest.CaptureFixture) -> None:
    """Test string representation of sigma config row contains row name."""
    print(config_sigma.default)
    captured = capsys.readouterr()
    assert "default" in captured.out


def test_sigma_config_index_type() -> None:
    """Test that sigma_config.index is a pandas Index."""
    assert isinstance(config_sigma.index, pd.Index)


def test_default_index_init() -> None:
    """Test that default_index is initialized to 'default'."""
    assert config_sigma.default_index == "default"


def test_default_sigmas_not_empty() -> None:
    """Test that default sigmas are not empty."""
    assert not config_sigma.default.empty


def test_set_new_default_index() -> None:
    """Test setting a new default index."""
    config_sigma.append("new_default_index")
    config_sigma.default_index = "new_default_index"
    assert config_sigma.default_index == "new_default_index"


def test_set_new_default_index_invalid() -> None:
    """Test setting an invalid default index raises ValueError."""
    with pytest.raises(ValueError):
        config_sigma.default_index = "value_not_in_index"


def test_default_index_delete() -> None:
    """Test deleting the default index raises ValueError."""
    with pytest.raises(ValueError):
        del config_sigma.default


def test_new_default_index_delete() -> None:
    """Test deleting a new default index resets to 'default'."""
    config_sigma.append("new_default_index_2")
    config_sigma.default_index = "new_default_index_2"
    assert config_sigma.default_index == "new_default_index_2"

    del config_sigma.new_default_index_2
    assert config_sigma.default_index == "default"


def test_delete_default_index_attr() -> None:
    """Test deleting a default index property raise warning and set default index to 'default'."""
    config_sigma.append("new_default_index_3")
    config_sigma.default_index = "new_default_index_3"
    assert config_sigma.default_index == "new_default_index_3"

    with pytest.warns(DefaultIndexWarning):
        del config_sigma.default_index

    assert config_sigma.default_index == "default"


def test_getattr_getitem() -> None:
    """Test dot notation and slice notation returns refernece for the same object."""
    assert config_sigma["default"] is config_sigma.default


def test_getitem_invalid() -> None:
    """Test slice invalid value raises attribute error."""
    with pytest.raises(AttributeError):
        config_sigma["invalid_name"]


def test_sigma_cofig_columns(sigma_columns: tuple[str]) -> None:
    """Test that all sigma columns are present in the dataframe."""
    for col in sigma_columns:
        assert col in config_sigma.columns


def test_sigma_cofig_columns_type(sigma_columns: tuple[str]) -> None:
    """Test that all sigma columns are of type float."""
    for col in sigma_columns:
        assert config_sigma[col].dtype == float


def test_append_invalid_name() -> None:
    """Test appending with an invalid name raises ValueError."""
    with pytest.raises(ValueError):
        config_sigma.append("123 Not identifier")


def test_append_full_row(
    sigma_columns: tuple[str], angle_units: tuple[str], rho: dict[str, float]
) -> None:
    """Test appending a full row with all columns."""
    new_row = {
        "stn_sh": 1,
        "trg_sh": 0,
        "ssd": 1,
        "shd": 0,
        "svd": 1,
        "sdx": 0,
        "sdy": 1,
        "sdz": 0,
        "sa": 1,
        "shz": 0,
        "svz": 1,
        "svh": 0,
        "sx": -1,
        "sy": 0,
        "sz": 1,
    }

    for unit in angle_units:
        new_row_name = f"new_row_{unit}"
        config_sigma.append(name=new_row_name, angle_unit=unit, **new_row)

        assert new_row_name in config_sigma.index

        for col in sigma_columns:
            if col not in ["sa", "shz", "svz", "svh"]:
                assert config_sigma[new_row_name][col] == new_row[col]
            else:
                assert config_sigma[new_row_name][col] == new_row[col] / rho[unit]


def test_append_incomplete_row(
    sigma_columns: tuple[str], angle_units: tuple[str], rho: dict[str, float]
) -> None:
    """Test appending a row with missing columns uses defaults."""
    incomplete_row = {
        "stn_sh": None,
        "trg_sh": 2,
        "shd": None,
        "svd": 2,
        "sdx": 2,
        "sdy": None,
        "sdz": 2,
        "sa": 2,
        "shz": 2,
        "svz": None,
        "sy": 2,
        "sz": None,
    }
    missing_columns = ("stn_sh", "ssd", "shd", "sdy", "svz", "svh", "sx", "sz")

    for unit in angle_units:
        new_row_name = f"incomplete_row_{unit}"
        config_sigma.append(new_row_name, angle_unit=unit, **incomplete_row)

        assert new_row_name in config_sigma.index

        for col in sigma_columns:
            if col in missing_columns:
                assert config_sigma[new_row_name][col] == config_sigma.default[col]
            elif col not in ["sa", "shz", "svz", "svh"]:
                assert config_sigma[new_row_name][col] == incomplete_row[col]
            else:
                assert (
                    config_sigma[new_row_name][col] == incomplete_row[col] / rho[unit]
                )


def test_append_without_name() -> None:
    """Test appending without a name auto-generates an index name."""
    n_index_before = len(config_sigma.index)
    config_sigma.append()
    n_index_after = len(config_sigma.index)

    assert n_index_after == n_index_before + 1
    assert f"index_{n_index_before - 1}" in config_sigma.index


def test_append_duplicate_name() -> None:
    """Test appending a duplicate name raises IndexError."""
    config_sigma.append("first_occurence")
    with pytest.raises(IndexError):
        config_sigma.append("first_occurence")


def test_append_invalid_values() -> None:
    """Test appending with invalid values raises ValueError."""
    invalid_row = {
        "trg_sh": -1,
        "shd": "Invalid_type",
        "sdy": -10,
        "sy": -2,
        "invalid_key": 2,
    }
    with pytest.raises(ValueError):
        config_sigma.append("invalid_row", angle_unit="rad", **invalid_row)


def test_display(sigma_columns: tuple[str], angle_units: tuple[str]) -> None:
    """Test display method returns correct values."""
    to_display = {
        "stn_sh": 1,
        "trg_sh": 0,
        "ssd": 1,
        "shd": 0,
        "svd": 1,
        "sdx": 0,
        "sdy": 1,
        "sdz": 0,
        "sa": 1,
        "shz": 0,
        "svz": 1,
        "svh": 0,
        "sx": -1,
        "sy": 0,
        "sz": 1,
    }

    for unit in angle_units:
        new_row_name = f"to_display_{unit}"
        config_sigma.append(new_row_name, angle_unit=unit, **to_display)
        displayed = config_sigma.display(angle_unit=unit)

        for col in sigma_columns:
            assert np.round(displayed.at[new_row_name, col], 15) == to_display[col]


def test_get_row(sigma_columns: tuple[str], angle_units: tuple[str]) -> None:
    """Test get_row returns correct values."""
    get_row = {
        "stn_sh": 1,
        "trg_sh": 0,
        "ssd": 1,
        "shd": 0,
        "svd": 1,
        "sdx": 0,
        "sdy": 1,
        "sdz": 0,
        "sa": 1,
        "shz": 0,
        "svz": 1,
        "svh": 0,
        "sx": -1,
        "sy": 0,
        "sz": 1,
    }

    for unit in angle_units:
        new_row_name = f"get_row_{unit}"
        config_sigma.append(new_row_name, angle_unit=unit, **get_row)

        row = config_sigma.get_row(new_row_name, angle_unit=unit)
        for col in sigma_columns:
            assert np.round(row[col], 15) == get_row[col]


def test_get_row_not_exists() -> None:
    """Test get_row with non-existent index raises IndexError."""
    with pytest.raises(IndexError):
        config_sigma.get_row("Index_not_exists")


def test_get_row_attr_get_row_item() -> None:
    """Test dot notation and slice notation returns the same values."""
    assert config_sigma.default["stn_sh"] == config_sigma.default.stn_sh
    assert config_sigma.default["trg_sh"] == config_sigma.default.trg_sh
    assert config_sigma.default["ssd"] == config_sigma.default.ssd
    assert config_sigma.default["shd"] == config_sigma.default.shd
    assert config_sigma.default["svd"] == config_sigma.default.svd
    assert config_sigma.default["sdx"] == config_sigma.default.sdx
    assert config_sigma.default["sdy"] == config_sigma.default.sdy
    assert config_sigma.default["sdz"] == config_sigma.default.sdz
    assert config_sigma.default["sa"] == config_sigma.default.sa
    assert config_sigma.default["shz"] == config_sigma.default.shz
    assert config_sigma.default["svz"] == config_sigma.default.svz
    assert config_sigma.default["svh"] == config_sigma.default.svh
    assert config_sigma.default["sx"] == config_sigma.default.sx
    assert config_sigma.default["sy"] == config_sigma.default.sy
    assert config_sigma.default["sz"] == config_sigma.default.sz


def test_field_setter() -> None:
    """Test setting fields via attribute assignment."""
    config_sigma.default.stn_sh = 20
    assert config_sigma.default.stn_sh == 20

    config_sigma.default.sx = -1
    assert config_sigma.default.sx == -1

    config_sigma.default.shz = 20
    assert config_sigma.default.shz == 20


def test_field_set_method(angle_units: tuple[str], rho: dict[str, float]) -> None:
    """Test setting fields via set() method with angle conversions."""

    for unit in angle_units:
        # Do not convert for distances
        config_sigma.default.set("shd", 20, angle_unit=unit)
        assert config_sigma.default.shd == 20
        # Enable -1 for control points
        config_sigma.default.set("sy", -1, angle_unit=unit)
        assert config_sigma.default.sy == -1
        # Do conversion for angles
        config_sigma.default.set("svz", 20, angle_unit=unit)
        assert config_sigma.default.svz == 20 / rho[unit]


def test_field_setter_invalid() -> None:
    """Test setting invalid values via attribute assignment raises ValueError."""
    with pytest.raises(ValueError):
        config_sigma.default.stn_sh = -1
    with pytest.raises(ValueError):
        config_sigma.default.sx = -2
    with pytest.raises(ValueError):
        config_sigma.default.shz = -1


def test_field_set_method_invalid(angle_units: tuple[str]) -> None:
    """Test setting invalid values via set() method raises ValueError."""
    for angle_unit in angle_units:
        with pytest.raises(ValueError):
            config_sigma.default.set("shd", -2, angle_unit=angle_unit)
        with pytest.raises(ValueError):
            config_sigma.default.set("sy", -2, angle_unit=angle_unit)
        with pytest.raises(ValueError):
            config_sigma.default.set("svz", -2, angle_unit=angle_unit)


def test_append_changed_default() -> None:
    """Test appending a row after changing the default uses changed default."""
    config_sigma.default.set("ssd", 20)
    config_sigma.append("changed_default", sdz=5)

    assert config_sigma.changed_default.sdz == 5
    assert config_sigma.changed_default.ssd == 20


def test_restore_default(sigma_columns: tuple[str]) -> None:
    """Test restore default sigma values restores from constants module."""
    config_sigma.default.trg_sh = 30
    config_sigma.default.sdx = 15
    config_sigma.default.sa = 7
    config_sigma.default.svh = 3
    config_sigma.default.sy = 2
    config_sigma.default.sz = -1

    assert config_sigma.default.trg_sh == 30
    assert config_sigma.default.sdx == 15
    assert config_sigma.default.sa == 7
    assert config_sigma.default.svh == 3
    assert config_sigma.default.sy == 2
    assert config_sigma.default.sz == -1

    config_sigma.restore_default()

    for col in sigma_columns:
        assert config_sigma.default[col] == DEFAULT_CONFIG_SIGMA[col]


def test_get(angle_units: tuple[str]) -> None:
    """Test get() method returns correct value with angle conversion."""
    for unit in angle_units:
        # Do not convert for distances
        config_sigma.default.set("shd", 50, angle_unit=unit)
        assert config_sigma.default.get("shd", angle_unit=unit) == 50
        # Enable -1 for control points
        config_sigma.default.set("sy", -1, angle_unit=unit)
        assert config_sigma.default.get("sy", angle_unit=unit) == -1
        # Do conversion for angles
        config_sigma.default.set("svz", 50, angle_unit=unit)
        assert config_sigma.default.get("svz", angle_unit=unit) == 50


def test_get_invalid_key() -> None:
    """Test get() raise value error on calling invalid field."""
    with pytest.raises(AttributeError):
        config_sigma.default.get("Invalid_key")
