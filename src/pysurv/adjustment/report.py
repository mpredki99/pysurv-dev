# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.


class Report:
    """Class for representing adjustment results."""

    def __init__(self, results: dict) -> None:
        self._results = results

    def __str__(self):
        return f"{self._results}"
