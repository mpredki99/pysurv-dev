# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

from abc import ABC, abstractmethod


class MatrixXYWsWStrategy(ABC):
    """
    Abstract base class for matrix construction strategies for least squares adjustment.

    This class defines the interface for strategies that build the design matrix (X),
    observation vector (Y), weight matrix for observations (W), and weight matrix for coordinates (sW)
    used in surveying network adjustment computations.

    Subclasses must implement the `xyw_constructor` and `sw_constructor` properties, which provide
    access to the respective matrix constructors.
    """

    @property
    @abstractmethod
    def xyw_constructor(self):
        """Returns constructor for X, Y, W matrices."""
        pass

    @property
    @abstractmethod
    def sw_constructor(self):
        """Returns constructor for sW matrix."""
        pass
