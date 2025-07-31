# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

from abc import ABC, abstractmethod
from inspect import signature

import numpy as np

from pysurv.data.dataset import Dataset
from pysurv.validators._validators import validate_method

from . import robust
from .matrix_constructors.indexer_matrix_x import IndexerMatrixX
from .matrix_constructors.strategy_matrix_xyw_sw_factory import get_strategy


class Matrices(ABC):
    """
    Class for constructing and managing matrices required for least squares adjustment.

    This class perform lazy builds and stores the design matrix (X), observation vector (Y),
    weight matrix for observations (W), weight matrix for coordinates (sW),
    inner constraints matrix (R), and control points corrections matrix (sX)
    for least squares computations in surveying network adjustment.
    """

    def __init__(
        self,
        dataset: Dataset,
        method: str = "weighted",
        tuning_constants: dict | None = None,
        free_adjustment: str | None = None,
        free_tuning_constants: dict | None = None,
        default_sigmas_index: str | None = None,
        build_strategy: str | None = None,
    ):
        self._dataset = dataset

        self._method = validate_method(method)
        self._tuning_constants = self._get_tuning_constants(
            tuning_constants, self._method
        )

        if free_adjustment is None:
            self._free_adjustment = None
        else:
            self._free_adjustment = validate_method(free_adjustment)
        self._free_tuning_constants = self._get_tuning_constants(
            free_tuning_constants, self._free_adjustment
        )

        self._indexer = None
        self._X = None
        self._Y = None
        self._W = None
        self._sW = None

        self._inner_constraints = None
        self._R = None
        self._sX = None

        self._k = None

        self._build_xyw_sw_strategy = get_strategy(
            self._dataset,
            self.indexer,
            default_sigmas_index,
            name=build_strategy,
        )

        self._hz_first_occurence = self._get_hz_first_occurence()
        if "hz" in self._dataset.measurements.angular_measurement_columns:
            self._update_stations_orientation()

    @property
    def method(self) -> str:
        """Return method used for weight matrix reweight."""
        return self._method

    @method.setter
    def method(self, value: str) -> None:
        """Set method used for weight matrix reweight."""
        self._method = validate_method(value)
        self._tuning_constants = self._get_tuning_constants(
            self._tuning_constants, self._method
        )

    @property
    def tuning_constants(self) -> dict:
        """Return tuning constants used for weight matrix reweight."""
        return self._tuning_constants

    @tuning_constants.setter
    def tuning_constants(self, value: dict | None) -> None:
        """Set tuning constants used for weight matrix reweight."""
        self._tuning_constants = self._get_tuning_constants(value, self._method)

    @property
    def free_adjustment(self) -> str:
        """Return method used for inner constraint matrix reweight."""
        return self._free_adjustment

    @free_adjustment.setter
    def free_adjustment(self, value: str | None) -> None:
        """Set method used for inner constraint matrix reweight."""
        if value is None:
            self._free_adjustment = None
        else:
            self._free_adjustment = validate_method(value)
        self._free_tuning_constants = self._get_tuning_constants(
            self._free_tuning_constants, self._free_adjustment
        )
        self._k = self._get_degrees_of_freedom()

    @property
    def free_tuning_constants(self) -> dict:
        """Return tuning constants used for inner constraint matrix reweight."""
        return self._free_tuning_constants

    @free_tuning_constants.setter
    def free_tuning_constants(self, value: dict | None) -> None:
        """Return tuning constants used for inner constraint matrix reweight."""
        self._free_tuning_constants = self._get_tuning_constants(
            value, self._free_adjustment
        )

    @property
    def calculate_weights(self) -> bool:
        """Return whether weight matrix is calculated for the adjustment."""
        return self._method != "ordinary"

    @property
    def apply_inner_constraints(self) -> bool:
        """Return whether inner constraints are applied for the adjustment."""
        return self._free_adjustment not in {None, "ordinary"}

    @property
    def indexer(self):
        if self._indexer is None:
            self._indexer = IndexerMatrixX(self._dataset)
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
        if self.calculate_weights and self._W is None:
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
        if self._free_adjustment is None and self._sX is None:
            self._build_sx_matrix()
        return self._sX

    @property
    def matrix_sW(self) -> np.ndarray:
        """Return the the control point weight matrix (sW) for least squares adjustment."""
        if self._free_adjustment != "ordinary" and self._sW is None:
            self._build_sw_matrix()
        return self._sW

    @property
    def degrees_of_freedom(self) -> int:
        """Return degrees of freedom of the system."""
        if self._k is None:
            self._k = self._get_degrees_of_freedom()
        return self._k

    def _get_degrees_of_freedom(self) -> int:
        """Calculate and return degrees of freedom of the system."""
        n_constraints = len(self.matrix_R) if self.matrix_R is not None else 0
        n_measurements, n_unknowns = self.matrix_X.shape
        return n_measurements + n_constraints - n_unknowns

    def _get_tuning_constants(
        self, tuning_constants: dict | None, method: str | None
    ) -> dict | None:
        """Determine and return tuning constants used for updating weights."""
        if method in {None, "ordinary", "weighted"}:
            return

        func = getattr(robust, method)
        default_kwargs = self._get_kwargs(func)

        if not tuning_constants:
            tuning_constants = default_kwargs

        tuning_constants = {
            key: tuning_constants.get(key, default_kwargs[key])
            for key in default_kwargs.keys()
        }

        if method == "cra":
            tuning_constants["sigma_sq"] = 1
        elif method == "t":
            tuning_constants["k"] = self.degrees_of_freedom
        return tuning_constants

    def _get_kwargs(self, func) -> dict:
        """Get kwargs with default values of robust method used for updating weights."""
        sig = signature(func)
        return {
            key: value.default
            for key, value in sig.parameters.items()
            if isinstance(value.default, (int, float))
        }

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
    def _update_weights(
        self, matrix: np.ndarray, v: np.ndarray, func: callable, tuning_constants: dict
    ) -> None:
        """Update proper weights matrix."""
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
