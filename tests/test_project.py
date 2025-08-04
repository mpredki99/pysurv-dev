# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

from pysurv import Dataset, Project, project_factory


def test_projec_factory(valid_measurement_file: str, valid_control_file: str) -> None:
    """Test that project factory creates project instance."""
    project = project_factory.from_csv(valid_measurement_file, valid_control_file)

    assert isinstance(project, Project)


def test_project_adjust(adjustment_test_dataset: Dataset) -> None:
    """Test that project adjust method works properly."""
    project = Project(adjustment_test_dataset)

    project.adjust()
    assert project.adjustment.report is not None
