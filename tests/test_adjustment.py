# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

from pysurv import Adjustment
from pysurv import Dataset

from pysurv.adjustment import MethodManager, DenseMatrices, Solver, Report



def test_instantiation(adjustment_test_dataset: Dataset):
    """Test that all adjustment objects are initialized properly."""
    adjustment = Adjustment(adjustment_test_dataset)
    
    assert isinstance(adjustment, Adjustment)
    assert isinstance(adjustment.methods, MethodManager)
    assert isinstance(adjustment.matrices, DenseMatrices)
    assert isinstance(adjustment.solver, Solver)
    assert adjustment.report is None
    
    
def test_report_instantiation(adjustment_test_dataset: Dataset):
    """Test that report object is initialized properly after solve adjustment task."""
    adjustment = Adjustment(adjustment_test_dataset)
    adjustment.solver.solve()
    
    assert isinstance(adjustment.report, Report)
    