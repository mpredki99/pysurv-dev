# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.


class PySurvWarning(Warning):
    pass


class ReaderWarning(PySurvWarning):
    pass


class InvalidDataWarning(ReaderWarning):
    pass


class AdjustmentWarning(PySurvWarning):
    pass


class DefaultIndexWarning(AdjustmentWarning):
    pass


class SVDNotConvergeWarning(AdjustmentWarning):
    pass


class InvalidVarianceWarning(AdjustmentWarning):
    pass


class DataWarning(PySurvWarning):
    pass


class InvalidGeometryWarning(DataWarning):
    pass


class GeometryAssigningWarning(DataWarning):
    pass
