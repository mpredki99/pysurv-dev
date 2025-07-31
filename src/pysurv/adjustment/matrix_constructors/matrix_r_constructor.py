# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

import numpy as np

from pysurv.data.dataset import Dataset

from .._constants import INVALID_INDEX
from .indexer_matrix_x import IndexerMatrixX
from .matrix_base_constructors import InnerConstraintsConstructor


class MatrixRConstructor(InnerConstraintsConstructor):
    """
    Constructs the inner constraints matrix R and the list of constraint types for adjustment computations.

    This class builds the R matrix, which encodes the inner constraints (such as translation, rotation, and scale)
    required for least squares free adjustment of surveying control networks. The constraints are determined based
    on the measurement types and control point coordinate columns present in the dataset.
    """

    def __init__(
        self, dataset: Dataset, matrix_x_indexer: IndexerMatrixX, matrix_x_n_col: int
    ) -> None:
        super().__init__(dataset, matrix_x_indexer, matrix_x_n_col)

    def build(self):
        """Build and return the inner constraints matrix R and the list of constraint types."""
        meas_types = self._dataset.measurements.measurement_columns
        coord_columns = self._dataset.controls.coordinate_columns

        # Constranits determined based on presence of measurements and control points in dataset
        obs_conditions = {
            "1D": not ("x" in coord_columns and "y" in coord_columns)
            and "z" in coord_columns,
            "3D": all(coord in coord_columns for coord in ["x", "y", "z"]),
            "sd": "sd" in meas_types
            or ("hd" in meas_types and "vd" in meas_types)
            or ("hd" in meas_types and "vz" in meas_types)
            or ("vd" in meas_types and "vz" in meas_types)
            or ("hd" in meas_types and "vh" in meas_types)
            or ("vd" in meas_types and "vh" in meas_types),
            "hd": "hd" in meas_types,
            "vd": any(measurement in meas_types for measurement in ["vd", "dz"]),
            "vector": all(
                measurement in meas_types for measurement in ["dx", "dy", "dz"]
            ),
            "hz": "hz" in meas_types,
            "vz": "vz" in meas_types,
            "a": "a" in meas_types,
        }

        # Control points network constraints
        constraint_conditions = {
            "rotate": not obs_conditions["1D"]
            and not (obs_conditions["vector"] or obs_conditions["a"]),
            "1D_scale": obs_conditions["1D"] and not obs_conditions["vd"],
            "2D_scale": not obs_conditions["1D"]
            and not any(obs_conditions[key] for key in ["vector", "sd", "hd", "vz"]),
            "3D_scale": obs_conditions["3D"]
            and not any(obs_conditions[key] for key in ["vector", "sd"]),
        }

        R, inner_constraints = self._translation_constraints()

        # Add rotate condition
        if constraint_conditions["rotate"]:
            R.append(self._rotate_constraint())
            inner_constraints.append("rotate")

        # Add scale constraint
        if constraint_conditions["1D_scale"]:
            R.append(self._scale_constraint(["z"]))
            inner_constraints.append("1D scale")

        elif constraint_conditions["2D_scale"]:
            R.append(self._scale_constraint(["x", "y"]))
            inner_constraints.append("2D scale")

        elif constraint_conditions["3D_scale"]:
            R.append(self._scale_constraint(["x", "y", "z"]))
            inner_constraints.append("3D scale")

        R = np.vstack(R)
        R = np.nan_to_num(R, nan=0)
        return R, inner_constraints

    def _translation_constraints(self):
        """Return translation constraints matrix rows and their labels."""
        inner_constraints = []
        R = []

        coord_columns = self._dataset.controls.coordinate_columns
        for coord_label in coord_columns:
            translation_condition = np.zeros(self._matrix_x_n_col)

            mask = (
                self._matrix_x_indexer.coordinate_indices[coord_label] != INVALID_INDEX
            )
            coord_indices = self._matrix_x_indexer.coordinate_indices[coord_label][mask]

            translation_condition[coord_indices] = 1
            R.append(translation_condition)
            inner_constraints.append(f"translation {coord_label}")
        return R, inner_constraints

    def _rotate_constraint(self):
        """Return rotate constraint matrix row."""
        coords = self._dataset.controls[["x", "y"]]
        rotate_constraint = np.zeros(self._matrix_x_n_col)

        for idx_coord, values_coord, sign in [("x", "y", 1), ("y", "x", -1)]:
            mask = self._matrix_x_indexer.coordinate_indices[idx_coord] != INVALID_INDEX
            coord_indices = self._matrix_x_indexer.coordinate_indices[idx_coord][mask]
            rotate_constraint[coord_indices] = sign * coords[values_coord].values[mask]

        return rotate_constraint

    def _scale_constraint(self, coord_cols):
        """Return scale constraint matrix row."""
        coords = self._dataset.controls.coordinates[coord_cols]
        scale_constraint = np.zeros(self._matrix_x_n_col)

        for coord in coord_cols:
            mask = self._matrix_x_indexer.coordinate_indices[coord] != INVALID_INDEX
            coord_indices = self._matrix_x_indexer.coordinate_indices[coord][mask]
            scale_constraint[coord_indices] = coords[coord].values[mask]

        return scale_constraint
