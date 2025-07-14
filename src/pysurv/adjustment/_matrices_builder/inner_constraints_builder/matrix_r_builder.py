import numpy as np

from ...constants import INVALID_INDEX
from .inner_constraints_builder import InnerConstraintsBuilder


class MatrixRBuilder(InnerConstraintsBuilder):
    def __init__(self, dataset, matrix_x_indexer, matrix_x_n_col):
        super().__init__(dataset, matrix_x_indexer, matrix_x_n_col)

    def build(self):
        inner_constraints = []
        R = []

        meas_types = self._dataset.measurements.measurements_columns
        coord_columns = self._dataset.controls.coordinates_columns

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

        constraint_conditions = {
            "rotate": not obs_conditions["1D"]
            and not (obs_conditions["vector"] or obs_conditions["a"]),
            "1D_scale": obs_conditions["1D"] and not obs_conditions["vd"],
            "2D_scale": not obs_conditions["1D"]
            and not any(obs_conditions[key] for key in ["vector", "sd", "hd", "vz"]),
            "3D_scale": obs_conditions["3D"]
            and not any(obs_conditions[key] for key in ["vector", "sd"]),
        }

        for coord_label in coord_columns:
            translation_condition = np.zeros(self._matrix_x_n_col)

            mask = self._matrix_x_indexer.coordinates_indices[coord_label] != INVALID_INDEX
            coord_indices = self._matrix_x_indexer.coordinates_indices[coord_label][
                mask
            ]

            translation_condition[coord_indices] = 1
            R.append(translation_condition)
            inner_constraints.append(f"translation {coord_label}")

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

    def _rotate_constraint(self):
        coords = self._dataset.controls[["x", "y"]]
        rotate_constraint = np.zeros(self._matrix_x_n_col)

        for idx_coord, values_coord, sign in [("x", "y", 1), ("y", "x", -1)]:
            mask = self._matrix_x_indexer.coordinates_indices[idx_coord] != INVALID_INDEX
            coord_indices = self._matrix_x_indexer.coordinates_indices[idx_coord][mask]
            rotate_constraint[coord_indices] = sign * coords[values_coord].values[mask]

        return rotate_constraint

    def _scale_constraint(self, coord_cols):
        coords = self._parent.dataset.controls.coordinates[coord_cols]
        scale_constraint = np.zeros(self._matrix_x_n_col)

        for coord in coord_cols:
            mask = self._matrix_x_indexer.coordinates_indices[coord] != INVALID_INDEX
            coord_indices = self._matrix_x_indexer.coordinates_indices[coord][mask]
            scale_constraint[coord_indices] = coords[coord].values[mask]

        return scale_constraint
