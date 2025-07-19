# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

from abc import ABC, abstractmethod

from pysurv.data.dataset import Dataset

from ._matrix_x_indexer import MatrixXIndexer


class MatrixConstructor(ABC):
    """
    Abstract base class for constructing matrices for leas squares adjustment of
    surveying control network.

    This class provides the interface for building matrices from a given `Dataset`.
    Matrix_x_indexer is helper object that maps control point coordinate indices
    and station orinetation indices into matrix X columns.

    Subclasses should implement the `build` method to return the constructed matrix.
    """

    def __init__(self, dataset: Dataset, matrix_x_indexer: MatrixXIndexer) -> None:
        self._dataset = dataset
        self._matrix_x_indexer = matrix_x_indexer

    @abstractmethod
    def build(self):
        """Build and return the constructed matrix."""
        pass
