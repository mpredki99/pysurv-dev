from ..observation_equations import *

obs_eqations_adapter = {
    "sd": lambda value, coord_diff, matrix_x_col_indices: apply_3D(
        sd_obs_eq, value, coord_diff, matrix_x_col_indices
    ),
    "hd": lambda value, coord_diff, matrix_x_col_indices: apply_2D(
        hd_obs_eq, value, coord_diff, matrix_x_col_indices
    ),
    "vd": lambda value, coord_diff, matrix_x_col_indices: apply_1D(
        dz_obs_eq,
        value,
        coord_diff["dz"],
        matrix_x_col_indices["z_stn"],
        matrix_x_col_indices["z_trg"],
    ),
    "dx": lambda value, coord_diff, matrix_x_col_indices: apply_1D(
        dx_obs_eq,
        value,
        coord_diff["dx"],
        matrix_x_col_indices["x_stn"],
        matrix_x_col_indices["x_trg"],
    ),
    "dy": lambda value, coord_diff, matrix_x_col_indices: apply_1D(
        dy_obs_eq,
        value,
        coord_diff["dy"],
        matrix_x_col_indices["y_stn"],
        matrix_x_col_indices["y_trg"],
    ),
    "dz": lambda value, coord_diff, matrix_x_col_indices: apply_1D(
        dz_obs_eq,
        value,
        coord_diff["dz"],
        matrix_x_col_indices["z_stn"],
        matrix_x_col_indices["z_trg"],
    ),
    "a": lambda value, coord_diff, matrix_x_col_indices: apply_2D(
        a_obs_eq, value, coord_diff, matrix_x_col_indices
    ),
    "hz": lambda value, coord_diff, matrix_x_col_indices: (
        (
            matrix_x_col_indices["x_stn"],
            matrix_x_col_indices["y_stn"],
            matrix_x_col_indices["orientation_idx"],
            matrix_x_col_indices["x_trg"],
            matrix_x_col_indices["y_trg"],
        ),
        *hz_obs_eq(
            value, coord_diff["dx"], coord_diff["dy"], coord_diff["orientation"]
        ),
    ),
    "vz": lambda value, coord_diff, matrix_x_col_indices: apply_3D(
        vz_obs_eq, value, coord_diff, matrix_x_col_indices
    ),
    "vh": lambda value, coord_diff, matrix_x_col_indices: apply_3D(
        vh_obs_eq, value, coord_diff, matrix_x_col_indices
    ),
}


def apply_3D(func, meas_value, coord_diff, matrix_x_col_indices):
    dx, dy, dz = coord_diff["dx"], coord_diff["dy"], coord_diff["dz"]
    x_stn = matrix_x_col_indices["x_stn"]
    y_stn = matrix_x_col_indices["y_stn"]
    z_stn = matrix_x_col_indices["z_stn"]
    x_trg = matrix_x_col_indices["x_trg"]
    y_trg = matrix_x_col_indices["y_trg"]
    z_trg = matrix_x_col_indices["z_trg"]

    output_indices = x_stn, y_stn, z_stn, x_trg, y_trg, z_trg
    output_coefficients, free_term = func(meas_value, dx, dy, dz)
    return output_indices, output_coefficients, free_term


def apply_2D(func, meas_value, coord_diff, matrix_x_col_indices):
    dx, dy = coord_diff["dx"], coord_diff["dy"]
    x_stn = matrix_x_col_indices["x_stn"]
    y_stn = matrix_x_col_indices["y_stn"]
    x_trg = matrix_x_col_indices["x_trg"]
    y_trg = matrix_x_col_indices["y_trg"]

    output_indices = x_stn, y_stn, x_trg, y_trg
    output_coefficients, free_term = func(meas_value, dx, dy)
    return output_indices, output_coefficients, free_term


def apply_1D(func, meas_value, dz, z_stn, z_trg):
    output_indices = z_stn, z_trg
    output_coefficients, free_term = func(meas_value, dz)
    return output_indices, output_coefficients, free_term
