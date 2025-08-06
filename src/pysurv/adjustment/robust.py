# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

import numpy as np
from scipy.special import erf

__all__ = [
    "huber",
    "slope",
    "hampel",
    "danish",
    "epanechnikov",
    "tukey",
    "jacobi",
    "exponential",
    "cra",
    "error_func",
    "cauchy",
    "t",
    "chain_bell",
    "chain",
    "andrews",
    "wave",
    "half_wave",
    "wigner",
    "ellipse_curve",
    "trim",
]


# FUNCTIONS WITH TOLERANCE
def huber(v: np.ndarray, c: float = 1.345) -> np.ndarray:
    """Compute Huber M-estimator reweight coefficinets."""
    v = np.abs(v)
    return np.where(v > c, c / v, 1)


def slope(v: np.ndarray, c: float = 2, a: float = 2) -> np.ndarray:
    """Compute slope function reweight coefficients."""
    v = np.abs(v)
    weights = 1 + (c - v) / a
    return np.clip(weights, 0, 1)


def hampel(v: np.ndarray, a: float = 1.7, b: float = 3.4, c: float = 8.5) -> np.ndarray:
    """Compute Hampel M-estimator reweight coefficients."""
    v = np.abs(v)
    weights = np.ones_like(v)
    weights[v > a] = np.divide(a, v[v > a])
    weights[v > b] = np.divide(a, v[v > b]) * np.divide(c - v[v > b], c - b)
    weights[v > c] = 0
    return weights


def danish(v: np.ndarray, c: float = 2.5) -> np.ndarray:
    """Compute danish method reweight coefficients."""
    v = np.abs(v)
    return np.where(v > c, np.exp(-v / c), 1)


# BELL CURVES
def epanechnikov(v: np.ndarray, c: float = 3.674, n: float = 2.0) -> np.ndarray:
    """Compute Epanechnikov M-estimator reweight coefficients."""
    v = np.abs(v)
    return np.where(v <= c, 1 - (v / c) ** n, 0)


def tukey(v: np.ndarray, c: float = 4.685, n: float = 2.0) -> np.ndarray:
    """Compute Tukey M-estimator reweight coefficients."""
    v = np.abs(v)
    return np.where(v <= c, (1 - (v / c) ** n) ** n, 0)


def jacobi(v: np.ndarray, c: float = 4.687, n: float = 1.0) -> np.ndarray:
    """Compute Jacobi M-estimator reweight coefficients."""
    v = np.abs(v)
    return np.where(v <= c, (1 - (v / c) ** n) ** n * (1 + (v / c) ** n) ** n, 0)


def exponential(v: np.ndarray, c: float = 2.0, n: float = 2.0) -> np.ndarray:
    """Compute exponential function reweight coefficients."""
    v = np.abs(v)
    return np.exp(-((v / c) ** n))


def cra(v: np.ndarray, sigma_sq, c: float = 2.0, n: float = 2.0) -> np.ndarray:
    """Compute the Choice Rule of Alternative method reweight coefficients."""
    v = np.abs(v)
    return np.exp(-(v**n) / (sigma_sq * c))


def error_func(v: np.ndarray, c: float = 1.414, n: float = 2.0) -> np.ndarray:
    """Compute error function reweight coefficients."""
    v = np.abs(v)
    return 1 - erf((v / c) ** n)


def cauchy(v: np.ndarray, c: float = 2.385, n: float = 2.0) -> np.ndarray:
    """Compute Cauchy M-estimator reweight coefficients."""
    v = np.abs(v)
    return np.divide(1, 1 + (v / c) ** n)


def t(v: np.ndarray, k: int, c: float = 1.0, n: float = 2.0) -> np.ndarray:
    """Compute T distribution reweight coefficients."""
    v = np.abs(v)
    return np.power(1 + v**n / (c * k), -(k + 1) / 2)


def chain_bell(v: np.ndarray, c: float = 1.0, n: float = 1.0) -> np.ndarray:
    """Compute chain bell function reweight coefficients."""
    v = np.abs(v)
    return np.divide(1, np.cosh(np.divide(v**n * np.e, 2 * c)))


# TRIGONOMETRIC FUNCTIONS
def chain(v: np.ndarray, c: float = 1.0) -> np.ndarray:
    """Compute chain function reweight coefficients."""
    v = np.abs(v)
    weights = -np.cosh((v * np.e) / (2 * c)) + 2
    return np.where(weights < 0, 0, weights)


def andrews(v: np.ndarray, c: float = 4.207) -> np.ndarray:
    """Compute Andrews M-estimator reweight coefficients."""
    v = np.abs(v)
    return np.where(v <= c, np.sinc(v / c), 0)


def wave(v: np.ndarray, c: float = 2.5) -> np.ndarray:
    """Compute wave function reweight coefficients."""
    v = np.abs(v)
    return np.where(v <= c, np.divide(np.cos(v * np.pi / c) + 1, 2), 0)


def half_wave(v: np.ndarray, c: float = 2.5) -> np.ndarray:
    """Compute half-wave function reweight coefficients."""
    v = np.abs(v)
    return np.where(v <= c, np.cos(v * np.pi / (2 * c)), 0)


# OTHERS
def wigner(v: np.ndarray, c: float = 3.137) -> np.ndarray:
    """Compute Wigner M-estimator reweight coefficients."""
    v = np.abs(v)
    return np.where(v <= c, np.sqrt(1 - (v / c) ** 2), 0)


def ellipse_curve(v: np.ndarray, c: float = 2.5) -> np.ndarray:
    """Compute ellipse curve function reweight coefficients."""
    v = np.abs(v)
    c_2 = 2 * c
    weights = np.ones_like(v)
    mask1 = v <= c
    mask2 = (v > c) & (v <= c_2)
    weights[mask1] = np.divide(1 + np.sqrt(1 - (v[mask1] / c) ** 2), 2)
    weights[mask2] = np.divide(1 - np.sqrt(1 - ((v[mask2] - c_2) / c) ** 2), 2)
    weights[v > c_2] = 0
    return weights


def trim(v: np.ndarray, c: float = 2.5) -> np.ndarray:
    """Compute trimming function reweight coefficients."""
    v = np.abs(v)
    return np.where(v <= c, 1, 0)
