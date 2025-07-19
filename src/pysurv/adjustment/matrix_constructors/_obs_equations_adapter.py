# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

from typing import Callable, Tuple

from ..observation_equations import *

"""
This module provides an adapter for mapping measurement types to their corresponding
observation equation functions and matrix indexers for least squares adjustment.

It imports observation equation functions from the `observation_equations` module and
defines a dictionary, `obs_eqations_adapter`, which maps measurement type strings
(e.g., "sd", "hd", "vd", "dx", "dy", "dz", "a", "hz") to lambda functions. These
lambdas apply the appropriate observation equation and return the indices and values
needed to populate the design matrix (X) and observation vector (Y) for adjustment.

Functions and mappings in this module are used internally by matrix construction
strategies to build the system of equations for network adjustment.
"""


obs_eqations_adapter = {
    "sd": lambda value, coord_diff, matrix_x_col_indices: apply_3D(
        sd_obs_eq, value, coord_diff, matrix_x_col_indices
    ),
    "hd": lambda value, coord_diff, matrix_x_col_indices: apply_2D(
        hd_obs_eq, value, coord_diff, matrix_x_col_indices
    ),
    "vd": lambda value, coord_diff, matrix_x_col_indices: (
        (
            matrix_x_col_indices["z_stn"],
            matrix_x_col_indices["z_trg"],
        ),
        *vector_obs_eq(value, coord_diff["dz"]),
    ),
    "dx": lambda value, coord_diff, matrix_x_col_indices: (
        (
            matrix_x_col_indices["x_stn"],
            matrix_x_col_indices["x_trg"],
        ),
        *vector_obs_eq(value, coord_diff["dx"]),
    ),
    "dy": lambda value, coord_diff, matrix_x_col_indices: (
        (
            matrix_x_col_indices["y_stn"],
            matrix_x_col_indices["y_trg"],
        ),
        *vector_obs_eq(value, coord_diff["dy"]),
    ),
    "dz": lambda value, coord_diff, matrix_x_col_indices: (
        (
            matrix_x_col_indices["z_stn"],
            matrix_x_col_indices["z_trg"],
        ),
        *vector_obs_eq(value, coord_diff["dz"]),
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


def apply_3D(
    func: Callable, meas_value: float, coord_diff: dict, matrix_x_col_indices: dict
) -> Tuple[
    Tuple[int, int, int, int, int, int],
    Tuple[float, float, float, float, float, float],
    float,
]:
    """Apply a 3D observation function."""
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


def apply_2D(
    func: Callable, meas_value: float, coord_diff: dict, matrix_x_col_indices: dict
) -> Tuple[Tuple[int, int, int, int], Tuple[float, float, float, float], float]:
    """Apply a 2D observation function."""
    dx, dy = coord_diff["dx"], coord_diff["dy"]
    x_stn = matrix_x_col_indices["x_stn"]
    y_stn = matrix_x_col_indices["y_stn"]
    x_trg = matrix_x_col_indices["x_trg"]
    y_trg = matrix_x_col_indices["y_trg"]

    output_indices = x_stn, y_stn, x_trg, y_trg
    output_coefficients, free_term = func(meas_value, dx, dy)
    return output_indices, output_coefficients, free_term
