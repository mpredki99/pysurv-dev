import numpy as np


class InnerConstraintsBuilder:
    def __init__(self, parent):
        self._parent = parent

    def build_r_matrix(self):
        inner_constraints = []
        R = []

        meas_types = self._parent.dataset.measurements.measurements_columns
        coord_columns = self._parent.dataset.controls.coordinates_columns

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
            translation_condition = np.zeros(self._parent.matrix_X.shape[1])

            mask = self._parent.cordinates_index_in_x_matrix[coord_label] != -1
            coord_indices = self._parent.cordinates_index_in_x_matrix[coord_label][mask]

            translation_condition[coord_indices] = 1
            R.append(translation_condition)
            inner_constraints.append(f"translation {coord_label}")

        # Add rotate condition
        if constraint_conditions["rotate"]:
            R.append(self._rotate_constraint())
            inner_constraints.append("rotate")

        # Add scale constraint
        # if constraint_conditions["1D_scale"]:
            R.append(self._scale_1D_constraint())
            inner_constraints.append("1D scale")

        # elif constraint_conditions["2D_scale"]:
            R.append(self._scale_2D_constraint())
            inner_constraints.append("2D scale")

        # elif constraint_conditions["3D_scale"]:
            R.append(self._scale_3D_constraint())
            inner_constraints.append("3D scale")

        R = np.vstack(R)
        R = np.nan_to_num(R, nan=0)
        return R, inner_constraints

    def _rotate_constraint(self):
        coords = self._parent.dataset.controls.coordinates

        rotate_constraint = np.zeros(self._parent.matrix_X.shape[1])

        mask_x = self._parent.cordinates_index_in_x_matrix.x != -1
        coord_indices_x = self._parent.cordinates_index_in_x_matrix.x[mask_x]

        mask_y = self._parent.cordinates_index_in_x_matrix.y != -1
        coord_indices_y = self._parent.cordinates_index_in_x_matrix.y[mask_y]

        rotate_constraint[coord_indices_x] = coords.y.values[mask_x]
        rotate_constraint[coord_indices_y] = -coords.x.values[mask_y]
        return rotate_constraint

    def _scale_1D_constraint(self):
        coords = self._parent.dataset.controls.coordinates

        scale_constraint = np.zeros(self._parent.matrix_X.shape[1])

        mask_z = self._parent.cordinates_index_in_x_matrix.z != -1
        coord_indices_z = self._parent.cordinates_index_in_x_matrix.z[mask_z]

        scale_constraint[coord_indices_z] = coords.z.values[mask_z]
        return scale_constraint

    def _scale_2D_constraint(self):
        coords = self._parent.dataset.controls.coordinates

        scale_constraint = np.zeros(self._parent.matrix_X.shape[1])

        mask_x = self._parent.cordinates_index_in_x_matrix.x != -1
        coord_indices_x = self._parent.cordinates_index_in_x_matrix.x[mask_x]
        mask_y = self._parent.cordinates_index_in_x_matrix.y != -1
        coord_indices_y = self._parent.cordinates_index_in_x_matrix.y[mask_y]

        scale_constraint[coord_indices_x] = coords.x.values[mask_x]
        scale_constraint[coord_indices_y] = coords.y.values[mask_y]
        return scale_constraint

    def _scale_3D_constraint(self):
        coords = self._parent.dataset.controls.coordinates

        scale_constraint = np.zeros(self._parent.matrix_X.shape[1])

        mask_x = self._parent.cordinates_index_in_x_matrix.x != -1
        coord_indices_x = self._parent.cordinates_index_in_x_matrix.x[mask_x]
        mask_y = self._parent.cordinates_index_in_x_matrix.y != -1
        coord_indices_y = self._parent.cordinates_index_in_x_matrix.y[mask_y]
        mask_z = self._parent.cordinates_index_in_x_matrix.z != -1
        coord_indices_z = self._parent.cordinates_index_in_x_matrix.z[mask_z]

        scale_constraint[coord_indices_x] = coords.x.values[mask_x]
        scale_constraint[coord_indices_y] = coords.y.values[mask_y]
        scale_constraint[coord_indices_z] = coords.z.values[mask_z]
        return scale_constraint
