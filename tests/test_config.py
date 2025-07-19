# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

import pytest

from pysurv import config
from pysurv.adjustment.sigma_config import SigmaConfig
from pysurv.exceptions import InvalidAngleUnitError


def test_singleton() -> None:
    """Test that config is a singleton."""
    config1 = config
    config2 = config
    assert config1 is config2


def test_angle_unit_type() -> None:
    """Test that angle_unit is a string."""
    assert isinstance(config.angle_unit, str)


def test_angle_unit_setter_valid() -> None:
    """Test setting valid angle units."""
    config.angle_unit = "deg"
    assert config.angle_unit == "deg"

    config.angle_unit = "gon"
    assert config.angle_unit == "gon"

    config.angle_unit = "grad"
    assert config.angle_unit == "grad"

    config.angle_unit = "rad"
    assert config.angle_unit == "rad"


def test_angle_unit_setter_none() -> None:
    """Test setting angle_unit to None retains current value."""
    config.angle_unit = "grad"
    config.angle_unit = None
    assert config.angle_unit == "grad"

    config.angle_unit = "rad"
    config.angle_unit = None
    assert config.angle_unit == "rad"


def test_angle_unit_setter_invalid() -> None:
    """Test setting invalid angle unit raises error."""
    with pytest.raises(InvalidAngleUnitError):
        config.angle_unit = "invalid_type"


def test_sigma_config_type() -> None:
    """Test that sigma_config is of type SigmaConfig."""
    assert isinstance(config.sigma_config, SigmaConfig)


def test_config_string(capsys: pytest.CaptureFixture) -> None:
    """Test string representation of config contains 'CONFIG'."""
    print(config)
    captured = capsys.readouterr()
    assert "CONFIG" in captured.out
