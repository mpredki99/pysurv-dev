# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

from pysurv.basic.basic import to_rad

MEMORY_THRESHOLD_GB = 1.0  # Threshold for switching to memory-safe strategy

INVALID_INDEX = -1  # Value used for invalid index in matrix constructors

DEFAULT_CONFIG_SIGMA = {
    "stn_sh": 0.01,  # Station height sigma
    "trg_sh": 0.01,  # Target height sigma
    "ssd": 0.01,  # Slope distance sigma
    "shd": 0.01,  # Horizontal distance sigma
    "svd": 0.01,  # Vertical distance sigma
    "sdx": 0.001,  # X coordinate difference sigma
    "sdy": 0.001,  # Y coordinate difference sigma
    "sdz": 0.001,  # Z coordinate difference sigma
    "sa": to_rad(0.0020, unit="grad"),  # Azimuth sigma (in radians)
    "shz": to_rad(0.0020, unit="grad"),  # Horizontal angle sigma (in radians)
    "svz": to_rad(0.0020, unit="grad"),  # Vertical angle sigma (in radians)
    "svh": to_rad(0.0020, unit="grad"),  # Vertical angle sigma (in radians)
    "sx": 0.01,  # X coordinate sigma
    "sy": 0.01,  # Y coordinate sigma
    "sz": 0.01,  # Z coordinate sigma
}

DEFAULT_CONFIG_SOLVER = {
    "threshold": 0.001,  # Max value of coordinate increments for breaking the iteration process
    "max_iter": 100,  # Maximum number of iterations
}
