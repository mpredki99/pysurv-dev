# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

import os
from typing import Any

from pysurv.data.dataset import Dataset

from .project import Project


def from_csv(
    measurements_file_path: str | os.PathLike,
    controls_file_path: str | os.PathLike,
    validation_mode: str | None = "raise",
    angle_unit: str | None = None,
    swap_xy: bool = False,
    delimiter: str | None = None,
    decimal: str = ".",
    crs: Any = None,
):
    """Create a Project from CSV files."""
    dataset = Dataset.from_csv(
        measurements_file_path=measurements_file_path,
        controls_file_path=controls_file_path,
        validation_mode=validation_mode,
        angle_unit=angle_unit,
        swap_xy=swap_xy,
        delimiter=delimiter,
        decimal=decimal,
        crs=crs,
    )
    return Project(dataset)
