# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.


class PySurvException(Exception):
    pass


class InvalidAngleUnitError(PySurvException):
    pass


class ReaderException(PySurvException):
    pass


class AdjustmentException(PySurvException):
    pass


class MissingMandatoryColumnsError(ReaderException):
    pass


class InvalidDataError(ReaderException):
    pass


class EmptyDatasetError(ReaderException):
    pass


class InvalidMethodError(AdjustmentException):
    pass
