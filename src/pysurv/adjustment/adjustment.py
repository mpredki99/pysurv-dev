# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

from .lsq_solver import LSQSolver
from .report import Report


class Adjustment:
    """Class for running least squares adjustment and show results."""
    def __init__(
        self,
        solver: LSQSolver,
    ) -> None:
        self.solver = solver
        self._report = None

    @property
    def report(self):
        """Return the adjustment report."""
        if self._report is None:
            self._report = Report(self.solver.results)
        return self._report

    def adjust(self):
        """Run the adjustment process."""
        self.solver.solve()
