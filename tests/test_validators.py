# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

import pytest

from pysurv import config
from pysurv.exceptions import InvalidAngleUnitError
from pysurv.validators import validate_angle_unit, validate_sigma


def test_validate_angle_unit_valid() -> None:
    """Test valid angle units are accepted."""
    angle_units = ["rad", "grad", "gon", "deg"]
    for angle_unit in angle_units:
        validated = validate_angle_unit(angle_unit)
        assert validated == angle_unit


def test_validate_angle_unit_none() -> None:
    """Test None returns angle unit from config."""
    validated = validate_angle_unit(None)
    assert validated == config.angle_unit


def test_validate_angle_unit_invalid() -> None:
    """Test invalid angle unit raises error."""
    with pytest.raises(InvalidAngleUnitError):
        validate_angle_unit("Invalid_value")


def test_validate_sigma_zero() -> None:
    """Test sigma value zero is valid."""
    validated = validate_sigma(0)
    assert validated == 0


def test_validate_sigma_positive() -> None:
    """Test positive sigma value is valid."""
    validated = validate_sigma(1)
    assert validated == 1


def test_validate_sigma_negative() -> None:
    """Test negative sigma value raises error."""
    with pytest.raises(ValueError):
        validate_sigma(-1)


def test_validate_sigma_enable_minus_one() -> None:
    """Test -1 is valid when enable_minus_one is True."""
    validated = validate_sigma(-1, enable_minus_one=True)
    assert validated == -1


def test_validate_sigma_custom_error_message() -> None:
    """Test custom error message is used on failure."""
    with pytest.raises(ValueError) as e:
        validate_sigma(-1, error_message="Test message")
        assert "Test message" in str(e.value)
