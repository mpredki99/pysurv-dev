# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

import numpy as np

from pysurv.basic import azimuth, from_rad, to_rad


def test_angles_to_rad() -> None:
    """Test conversion of angles to radians."""
    angles = {"rad": np.pi, "grad": 200, "gon": 200, "deg": 180}

    for unit, angle in angles.items():
        value: float = to_rad(angle, unit=unit)
        assert value == np.pi


def test_angles_from_rad() -> None:
    """Test conversion of angles from radians to other angle units."""
    angles = {"rad": np.pi, "grad": 200, "gon": 200, "deg": 180}

    for unit, angle in angles.items():
        value: float = from_rad(np.pi, unit=unit)
        assert value == angle


def test_azimuth_overlaping_points() -> None:
    """Test azimuth for overlapping points."""
    value: float = azimuth(0, 0, 0, 0)
    assert value == 0


def test_azimuth_north_direction() -> None:
    """Test azimuth for north direction."""
    value: float = azimuth(0, 0, 100, 0)
    assert value == 0


def test_azimuth_first_quarter() -> None:
    """Test azimuth in the first quarter."""
    value: float = azimuth(0, 0, 100, 100)
    assert value == np.pi / 4


def test_azimuth_east_direction() -> None:
    """Test azimuth for east direction."""
    value: float = azimuth(0, 0, 0, 100)
    assert value == np.pi / 2


def test_azimuth_second_quarter() -> None:
    """Test azimuth in the second quarter."""
    value: float = azimuth(0, 0, -100, 100)
    assert value == np.pi * 3 / 4


def test_azimuth_south_direction() -> None:
    """Test azimuth for south direction."""
    value: float = azimuth(0, 0, -100, 0)
    assert value == np.pi


def test_azimuth_third_quarter() -> None:
    """Test azimuth in the third quarter."""
    value: float = azimuth(0, 0, -100, -100)
    assert value == np.pi * 5 / 4


def test_azimuth_west_direction() -> None:
    """Test azimuth for west direction."""
    value: float = azimuth(0, 0, 0, -100)
    assert value == np.pi * 3 / 2


def test_azimuth_forth_quarter() -> None:
    """Test azimuth in the fourth quarter."""
    value: float = azimuth(0, 0, 100, -100)
    assert value == np.pi * 7 / 4
