# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from pysurv.data.dataset import Dataset

from .matrix_constructors.indexer_matrix_x import IndexerMatrixX
from .matrix_constructors.strategy_matrix_xyw_sw_factory import get_strategy
from .method_manager_adjustment import MethodManagerAdjustment


class Matrices(ABC):
    """
    Base class for constructing and managing matrices required for least squares adjustment.

    This class perform lazy builds and stores the design matrix (X), observation vector (Y),
    weight matrix for observations (W), weight matrix for coordinates (sW),
    inner constraints matrix (R), and control points corrections matrix (sX)
    for least squares computations in surveying network adjustment.
    """

    def __init__(
        self,
        dataset: Dataset,
        methods: MethodManagerAdjustment,
        config_sigma_index: str | None = None,
        build_strategy: str | None = None,
    ):
        self._dataset = dataset

        self._indexer = self._get_matrix_x_indexer()
        self._X = None
        self._Y = None
        self._W = None
        self._sW = None

        self._inner_constraints = None
        self._R = None
        self._sX = None

        self._k = None

        self._xyw_sw_init_strategy = get_strategy(
            self._dataset,
            self._indexer,
            config_sigma_index,
            name=build_strategy,
        )

        self._hz_first_occurence = None
        if "hz" in self._dataset.measurements.angular_measurement_columns:
            self._hz_first_occurence = self._get_hz_first_occurence()
            self._update_stations_orientation()

        self._methods = methods
        self._methods._inject_matrices(self)

    @property
    def methods(self):
        return self._methods

    @property
    def calculate_weights(self) -> bool:
        """Return whether weight matrix is calculated for the adjustment."""
        return self._methods.observations != "ordinary"

    @property
    def apply_inner_constraints(self) -> bool:
        """Return whether inner constraints are applied for the adjustment."""
        return self._methods.free_adjustment not in {None, "ordinary"}

    @property
    def indexer(self):
        return self._indexer

    @property
    def matrix_X(self) -> np.ndarray:
        """Return the design matrix (X) for least squares adjustment."""
        if self._X is None:
            self._build_xyw_matrices()
        return self._X

    @property
    def matrix_Y(self) -> np.ndarray:
        """Return the observation vector (Y) for least squares adjustment."""
        if self._Y is None:
            self._build_xyw_matrices()
        return self._Y

    @property
    def matrix_W(self) -> np.ndarray:
        """Return the observation weight matrix (W) for least squares adjustment."""
        if not self.calculate_weights:
            return

        if self._W is None:
            self._build_xyw_matrices()
        return self._W

    @property
    def inner_constraints(self) -> list:
        """Return the list of inner constraints used in the adjustment."""
        if not self.apply_inner_constraints:
            return

        if self._inner_constraints is None:
            self._build_inner_constraints_matrix()
        return self._inner_constraints

    @property
    def matrix_R(self) -> np.ndarray:
        """Return the inner constraints matrix (R) for least squares adjustment."""
        if not self.apply_inner_constraints:
            return

        if self._R is None:
            self._build_inner_constraints_matrix()
        return self._R

    @property
    def matrix_sX(self) -> np.ndarray:
        """Return the control point corrections matrix (sX) for least squares adjustment."""
        if self._methods.free_adjustment is not None:
            return

        if self._sX is None:
            self._build_sx_matrix()
        return self._sX

    @property
    def matrix_sW(self) -> np.ndarray:
        """Return the the control point weight matrix (sW) for least squares adjustment."""
        if self._methods.free_adjustment == "ordinary":
            return

        if self._sW is None:
            self._build_sw_matrix()
        return self._sW

    @property
    def degrees_of_freedom(self) -> int:
        """Return degrees of freedom of the system."""
        if self._k is None:
            self._refresh_degrees_of_freedom()
        return self._k

    def _get_matrix_x_indexer(self):
        """Return IndexerMatrixX object."""
        return IndexerMatrixX(self._dataset)

    def _refresh_degrees_of_freedom(self):
        """Set new value of degrees of freedom"""
        self._k = self._get_degrees_of_freedom()

    def _get_degrees_of_freedom(self) -> int:
        """Calculate and return degrees of freedom of the system."""
        n_constraints = len(self.matrix_R) if self.matrix_R is not None else 0
        n_measurements, n_unknowns = self.matrix_X.shape
        return n_measurements + n_constraints - n_unknowns

    def _get_hz_first_occurence(self) -> pd.Series:
        """Set the values of first occurrence of each 'hz' measurement for each station."""
        hz = self._dataset.measurements.hz.dropna()
        first_hz_occurence = hz.reset_index().drop_duplicates("stn_pk").index
        return hz.iloc[first_hz_occurence]

    def _update_stations_orientation(self) -> None:
        """Update the orientation constant for stations based on 'hz' measurements."""
        self._dataset.stations.append_orientation_constant(
            self._hz_first_occurence, self._dataset.controls
        )

    @abstractmethod
    def _build_xyw_matrices(self) -> None:
        """Build the X, Y, and W matrices for least squares adjustment."""
        pass

    @abstractmethod
    def _build_inner_constraints_matrix(self) -> None:
        """Build the R matrix and inner constraints list."""
        pass

    @abstractmethod
    def _build_sx_matrix(self):
        """Build the control point corrections matrix (sX)."""
        pass

    @abstractmethod
    def _build_sw_matrix(self):
        """Build the control point weights matrix (sW)."""
        pass

    @abstractmethod
    def update_xy_matrices(self) -> None:
        """Update the X and Y matrices for least squares adjustment."""
        pass

    @abstractmethod
    def update_w_matrix(self, v: np.ndarray) -> None:
        """Update observation weights matrix."""
        pass

    @abstractmethod
    def update_sw_matrix(self, v: np.ndarray) -> None:
        """Update control point weights matrix."""
        pass
