# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

from typing import Tuple

import numpy as np

from pysurv.basic.basic import azimuth

__all__ = [
    "sd_obs_eq",
    "hd_obs_eq",
    "vector_obs_eq",
    "a_obs_eq",
    "hz_obs_eq",
    "vz_obs_eq",
    "vh_obs_eq",
]


def sd_obs_eq(
    meas_sd: float, dx: float, dy: float, dz: float
) -> Tuple[Tuple[float, float, float, float, float, float], float]:
    """Observation equation for slope distance measurement."""
    distance = np.linalg.norm([dx, dy, dz])
    # Calculate the coefficients of the matrix X
    x_stn_coeff = -np.divide(dx, distance)
    y_stn_coeff = -np.divide(dy, distance)
    z_stn_coeff = -np.divide(dz, distance)
    x_trg_coeff = -x_stn_coeff
    y_trg_coeff = -y_stn_coeff
    z_trg_coeff = -z_stn_coeff
    # Calculate the value of the free term
    free_term = meas_sd - distance

    return (
        x_stn_coeff,
        y_stn_coeff,
        z_stn_coeff,
        x_trg_coeff,
        y_trg_coeff,
        z_trg_coeff,
    ), free_term


def hd_obs_eq(
    meas_hd: float, dx: float, dy: float
) -> Tuple[Tuple[float, float, float, float], float]:
    """Observation equation for horizontal distance measurement."""
    distance = np.linalg.norm([dx, dy])
    # Calculate the coefficients of the matrix X
    x_stn_coeff = -np.divide(dx, distance)
    y_stn_coeff = -np.divide(dy, distance)
    x_trg_coeff = -x_stn_coeff
    y_trg_coeff = -y_stn_coeff
    # Calculate the value of the free term
    free_term = meas_hd - distance

    return (x_stn_coeff, y_stn_coeff, x_trg_coeff, y_trg_coeff), free_term


def vector_obs_eq(
    meas_vector_component: float, coord_diff: float
) -> Tuple[Tuple[float, float], float]:
    """Observation equation for GNSS vector component."""
    # Calculate the coefficients of the matrix X
    stn_coeff = -1
    trg_coeff = 1
    # Calculate the value of the free term
    free_term = meas_vector_component - coord_diff

    return (stn_coeff, trg_coeff), free_term


def a_obs_eq(
    meas_a: float, dx: float, dy: float
) -> Tuple[Tuple[float, float, float, float], float]:
    """Observation equation for azimuth measurement."""
    distance_sq = np.linalg.norm([dx, dy]) ** 2
    # Calculate the coefficients of the matrix X
    x_stn_coeff = np.divide(dy, distance_sq)
    y_stn_coeff = -np.divide(dx, distance_sq)
    x_trg_coeff = -x_stn_coeff
    y_trg_coeff = -y_stn_coeff
    # Calculate the value of the free term
    free_term = meas_a - azimuth(0, 0, dx, dy)

    return (x_stn_coeff, y_stn_coeff, x_trg_coeff, y_trg_coeff), free_term


def hz_obs_eq(
    meas_hz: float, dx: float, dy: float, orientation: float
) -> Tuple[Tuple[float, float, float, float, float], float]:
    # Predefine necessary values
    distance_sq = np.linalg.norm([dx, dy]) ** 2
    # Calculate the coefficients of the matrix X
    x_stn_coeff = np.divide(dy, distance_sq)
    y_stn_coeff = -np.divide(dx, distance_sq)
    orientation_coeff = -1
    x_trg_coeff = -x_stn_coeff
    y_trg_coeff = y_stn_coeff
    # Calculate the value of the free term
    free_term = meas_hz - azimuth(0, 0, dx, dy) + orientation
    # Reduce the value of the free term if necessary
    normalized_free_term = np.arctan2(np.sin(free_term), np.cos(free_term))

    return (
        x_stn_coeff,
        y_stn_coeff,
        orientation_coeff,
        x_trg_coeff,
        y_trg_coeff,
    ), normalized_free_term


def vz_obs_eq(
    meas_vz: float, dx: float, dy: float, dz: float
) -> Tuple[Tuple[float, float, float, float, float, float], float]:
    """Observation equation for zenith angle measurement."""
    distance_hd = np.linalg.norm([dx, dy])
    distance_sd_sq = np.linalg.norm([dx, dy, dz]) ** 2
    # Calculate the coefficients of the matrix X
    x_stn_coeff = -np.divide(dx * dz, distance_hd * distance_sd_sq)
    y_stn_coeff = -np.divide(dy * dz, distance_hd * distance_sd_sq)
    z_stn_coeff = np.divide(distance_hd, distance_sd_sq)
    x_trg_coeff = -x_stn_coeff
    y_trg_coeff = -y_stn_coeff
    z_trg_coeff = -z_stn_coeff
    # Calculate the value of the free term
    free_term = meas_vz - np.arctan2(distance_hd, dz)

    return (
        x_stn_coeff,
        y_stn_coeff,
        z_stn_coeff,
        x_trg_coeff,
        y_trg_coeff,
        z_trg_coeff,
    ), free_term


def vh_obs_eq(
    meas_vh: float, dx: float, dy: float, dz: float
) -> Tuple[Tuple[float, float, float, float, float, float], float]:
    """Observation equation for vertical angle measurement."""
    distance_hd = np.linalg.norm([dx, dy])
    distance_sd_sq = np.linalg.norm([dx, dy, dz]) ** 2
    # Calculate the coefficients of the matrix X
    x_stn_coeff = np.divide(dx * dz, distance_hd * distance_sd_sq)
    y_stn_coeff = np.divide(dy * dz, distance_hd * distance_sd_sq)
    z_stn_coeff = -np.divide(distance_hd, distance_sd_sq)
    x_trg_coeff = -x_stn_coeff
    y_trg_coeff = -y_stn_coeff
    z_trg_coeff = -z_stn_coeff
    # Calculate the value of the free term
    free_term = np.array(meas_vh - np.arctan2(dz, distance_hd))

    return (
        x_stn_coeff,
        y_stn_coeff,
        z_stn_coeff,
        x_trg_coeff,
        y_trg_coeff,
        z_trg_coeff,
    ), free_term
