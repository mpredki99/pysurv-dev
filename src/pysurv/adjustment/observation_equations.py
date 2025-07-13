import numpy as np

from pysurv.basic.basic import azimuth

__all__ = [
    "sd_obs_eq",
    "hd_obs_eq",
    "dx_obs_eq",
    "dy_obs_eq",
    "dz_obs_eq",
    "a_obs_eq",
    "hz_obs_eq",
    "vz_obs_eq",
    "vh_obs_eq",
]


def sd_obs_eq(meas_sd, dx, dy, dz):
    # Predefine necessary values
    distance = np.linalg.norm([dx, dy, dz])
    # Calculate the coefficients of the matrix X
    x_stn = -np.divide(dx, distance)
    y_stn = -np.divide(dy, distance)
    z_stn = -np.divide(dz, distance)
    x_trg = -x_stn
    y_trg = -y_stn
    z_trg = -z_stn
    # Calculate the value of the free term
    free_term = meas_sd - distance

    return (x_stn, y_stn, z_stn, x_trg, y_trg, z_trg), free_term


def hd_obs_eq(meas_hd, dx, dy):
    # Predefine necessary values
    distance = np.linalg.norm([dx, dy])
    # Calculate the coefficients of the matrix X
    x_stn = -np.divide(dx, distance)
    y_stn = -np.divide(dy, distance)
    x_trg = -x_stn
    y_trg = -y_stn
    # Calculate the value of the free term
    free_term = meas_hd - distance

    return (x_stn, y_stn, x_trg, y_trg), free_term


def dx_obs_eq(meas_dx, dx):
    # Calculate the coefficients of the matrix X
    x_stn = -1
    x_trg = 1
    # Calculate the value of the free term
    free_term = meas_dx - dx

    return (x_stn, x_trg), free_term


def dy_obs_eq(meas_dy, dy):
    # Calculate the coefficients of the matrix X
    y_stn = -1
    y_trg = 1
    # Calculate the value of the free term
    free_term = meas_dy - dy

    return (y_stn, y_trg), free_term


def dz_obs_eq(meas_dz, dz):
    # Calculate the coefficients of the matrix X
    z_stn = -1
    z_trg = 1
    # Calculate the value of the free term
    free_term = meas_dz - dz

    return (z_stn, z_trg), free_term


def a_obs_eq(meas_a, dx, dy):
    # Predefine necessary values
    distance_sq = np.linalg.norm([dx, dy]) ** 2
    # Calculate the coefficients of the matrix X
    x_stn = np.divide(dy, distance_sq)
    y_stn = -np.divide(dx, distance_sq)
    x_trg = -x_stn
    y_trg = -y_stn
    # Calculate the value of the free term
    free_term = meas_a - azimuth(0, 0, dx, dy)

    return (x_stn, y_stn, x_trg, y_trg), free_term


def hz_obs_eq(meas_hz, dx, dy, orientation):
    # Predefine necessary values
    distance_sq = np.linalg.norm([dx, dy]) ** 2
    # Calculate the coefficients of the matrix X
    x_stn = np.divide(dy, distance_sq)
    y_stn = -np.divide(dx, distance_sq)
    orientation_stn = -1
    x_trg = -np.divide(dy, distance_sq)
    y_trg = np.divide(dx, distance_sq)
    # Calculate the value of the free term
    free_term = meas_hz - azimuth(0, 0, dx, dy) + orientation
    # Reduce the value of the free term if necessary
    normalized_free_term = np.arctan2(np.sin(free_term), np.cos(free_term))

    return (x_stn, y_stn, orientation_stn, x_trg, y_trg), normalized_free_term


def vz_obs_eq(meas_vz, dx, dy, dz):
    # Predefine necessary values
    distance_hd = np.linalg.norm([dx, dy])
    distance_sd_sq = np.linalg.norm([dx, dy, dz]) ** 2
    # Calculate the coefficients of the matrix X
    x_stn = -np.divide(dx * dz, distance_hd * distance_sd_sq)
    y_stn = -np.divide(dy * dz, distance_hd * distance_sd_sq)
    z_stn = np.divide(distance_hd, distance_sd_sq)
    x_trg = -x_stn
    y_trg = -y_stn
    z_trg = -z_stn
    # Calculate the value of the free term
    free_term = meas_vz - np.arctan2(distance_hd, dz)

    return (x_stn, y_stn, z_stn, x_trg, y_trg, z_trg), free_term


def vh_obs_eq(meas_vh, dx, dy, dz):
    # Predefine necessary values
    distance_hd = np.linalg.norm([dx, dy])
    distance_sd_sq = np.linalg.norm([dx, dy, dz]) ** 2
    # Calculate the coefficients of the matrix X
    x_stn = np.divide(dx * dz, distance_hd * distance_sd_sq)
    y_stn = np.divide(dy * dz, distance_hd * distance_sd_sq)
    z_stn = -np.divide(distance_hd, distance_sd_sq)
    x_trg = -x_stn
    y_trg = -y_stn
    z_trg = -z_stn
    # Calculate the value of the free term
    free_term = np.array(meas_vh - np.arctan2(dz, distance_hd))

    return (x_stn, y_stn, z_stn, x_trg, y_trg, z_trg), free_term
