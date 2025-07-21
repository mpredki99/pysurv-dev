# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

import numpy as np

from pysurv.data.dataset import Dataset

from .matrix_constructors._matrix_r_constructor import MatrixRConstructor
from .matrix_constructors._matrix_sx_constructor import MatrixSXConstructor
from .matrix_constructors._matrix_x_indexer import MatrixXIndexer
from .matrix_constructors._matrix_xyw_sw_strategy_factory import get_strategy


class LSQMatrices:
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
        calculate_weights: bool = True,
        default_sigmas_index: str | None = None,
        computations_priority: str | None = None,
    ):
        self._dataset = dataset
        self._calculate_weights = calculate_weights

        self._matrix_x_indexer = MatrixXIndexer(self._dataset)
        self._X = None
        self._Y = None
        self._W = None
        self._sW = None

        self._inner_constraints = None
        self._R = None
        self._sX = None

        self._matrices_xyw_sw_build_strategy = get_strategy(
            self._dataset,
            self._matrix_x_indexer,
            default_sigmas_index,
            name=computations_priority,
        )

        self._hz_first_occurence = None
        if "hz" in self.dataset.measurements.angular_measurement_columns:
            self._update_stations_orientation()

    @property
    def dataset(self) -> Dataset:
        """Return the dataset used for matrix construction."""
        return self._dataset

    @property
    def calculate_weights(self) -> bool:
        """Return whether weight matrix is calculated for the adjustment."""
        return self._calculate_weights

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
        if self._W is None and self._calculate_weights:
            self._build_xyw_matrices()
        return self._W

    @property
    def matrix_sW(self) -> np.ndarray:
        """Return the the control point weight matrix (sW) for least squares adjustment."""
        if self._sW is None:
            self._sW = self._matrices_xyw_sw_build_strategy.sw_constructor.build(
                self.matrix_X.shape[1]
            )
        return self._sW

    @property
    def inner_constraints(self) -> list:
        """Return the list of inner constraints used in the adjustment."""
        if self._inner_constraints is None:
            self._apply_inner_constraints()
        return self._inner_constraints

    @property
    def matrix_R(self) -> np.ndarray:
        """Return the inner constraints matrix (R) for least squares adjustment."""
        if self._R is None:
            self._apply_inner_constraints()
        return self._R

    @property
    def matrix_sX(self) -> np.ndarray:
        """Return the control point corrections matrix (sX) for least squares adjustment."""
        if self._sX is None:
            matrix_sx_builder = MatrixSXConstructor(
                self.dataset, self._matrix_x_indexer, self.matrix_X.shape[1]
            )
            self._sX = matrix_sx_builder.build()
        return self._sX

    def _set_hz_first_occurence(self) -> None:
        """Set the values of first occurrence of each 'hz' measurement for each station."""
        hz = self._dataset.measurements.hz.dropna()
        stn_pk = hz.reset_index()["stn_pk"]
        first_hz_occurence = stn_pk.drop_duplicates().index
        self._hz_first_occurence = hz.iloc[first_hz_occurence]

    def _build_xyw_matrices(self) -> None:
        """Build the X, Y, and W matrices for least squares adjustment."""
        self._X, self._Y, self._W = (
            self._matrices_xyw_sw_build_strategy.xyw_constructor.build(
                calculate_weights=self._calculate_weights
            )
        )

    def _apply_inner_constraints(self) -> None:
        """Build the R matrix and inner constraints list."""
        inner_constraints_builder = MatrixRConstructor(
            self.dataset, self._matrix_x_indexer, self.matrix_X.shape[1]
        )
        self._R, self._inner_constraints = inner_constraints_builder.build()

    def _update_stations_orientation(self) -> None:
        """Update the orientation constant for stations based on 'hz' measurements."""
        if self._hz_first_occurence is None:
            self._set_hz_first_occurence()

        self._dataset.stations.append_orientation_constant(
            self._hz_first_occurence, self.dataset.controls
        )

    def update_xy_matrices(self) -> None:
        """Update the X and Y matrices for least squares adjustment."""
        if "hz" in self.dataset.measurements.angular_measurements_columns:
            self._update_stations_orientation()
        self._X, self._Y, _ = (
            self._matrices_xyw_sw_build_strategy.xyw_constructor.build(
                calculate_weights=False
            )
        )

    def update_w_matrix(self):
        # TODO
        pass

    def update_sw_matrix(self):
        # TODO
        pass
