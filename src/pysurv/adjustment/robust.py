# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

import numpy as np
from scipy.special import erf

from pysurv.utils import apply_where, inf_to_zero

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


# FUNCTIONS WITH TOLERANCE -------------------------------
@inf_to_zero
def huber(v: np.ndarray, c: float = 1.345) -> np.ndarray:
    """Compute Huber M-estimator reweight coefficinets."""
    v = np.abs(v)
    formula = lambda x: c / x
    return apply_where(v, v > c, formula, np.ones_like)


@inf_to_zero
def slope(v: np.ndarray, c: float = 2, a: float = 2) -> np.ndarray:
    """Compute slope function reweight coefficients."""
    v = np.abs(v)
    coeff = 1 + (c - v) / a
    return np.clip(coeff, 0, 1)


@inf_to_zero
def hampel(v: np.ndarray, a: float = 1.7, b: float = 3.4, c: float = 8.5) -> np.ndarray:
    """Compute Hampel M-estimator reweight coefficients."""
    v = np.abs(v)
    coeff = np.ones_like(v)

    mask1 = (v > a) & (v <= b)
    mask2 = (v > b) & (v <= c)
    mask3 = v > c

    coeff[mask1] = np.divide(a, v[mask1])
    coeff[mask2] = np.divide(a, v[mask2]) * np.divide(c - v[mask2], c - b)
    coeff[mask3] = 0

    return coeff


@inf_to_zero
def danish(v: np.ndarray, c: float = 2.5) -> np.ndarray:
    """Compute danish method reweight coefficients."""
    v = np.abs(v)
    formula = lambda x: np.exp(-x / c)
    return apply_where(v, v > c, formula, np.ones_like)


# BELL CURVES ------------------------------------------------------------------
@inf_to_zero
def epanechnikov(v: np.ndarray, c: float = 3.674, n: float = 2.0) -> np.ndarray:
    """Compute Epanechnikov M-estimator reweight coefficients."""
    v = np.abs(v)
    formula = lambda x: 1 - (x / c) ** n
    return apply_where(v, v <= c, formula, np.zeros_like)


@inf_to_zero
def tukey(v: np.ndarray, c: float = 4.685, n: float = 2.0) -> np.ndarray:
    """Compute Tukey M-estimator reweight coefficients."""
    v = np.abs(v)
    formula = lambda x: (1 - (x / c) ** n) ** n
    return apply_where(v, v <= c, formula, np.zeros_like)


@inf_to_zero
def jacobi(v: np.ndarray, c: float = 4.687, n: float = 1.0) -> np.ndarray:
    """Compute Jacobi M-estimator reweight coefficients."""
    v = np.abs(v)
    formula = lambda x: (1 - (x / c) ** n) ** n * (1 + (x / c) ** n) ** n
    return apply_where(v, v <= c, formula, np.zeros_like)


@inf_to_zero
def exponential(v: np.ndarray, c: float = 2.0, n: float = 2.0) -> np.ndarray:
    """Compute exponential function reweight coefficients."""
    v = np.abs(v)
    return np.exp(-((v / c) ** n))


@inf_to_zero
def cra(v: np.ndarray, res_var: float, c: float = 2.0, n: float = 2.0) -> np.ndarray:
    """Compute the Choice Rule of Alternative method reweight coefficients."""
    v = np.abs(v)
    return np.exp(-(v**n) / (res_var * c))


@inf_to_zero
def error_func(v: np.ndarray, c: float = 1.414, n: float = 2.0) -> np.ndarray:
    """Compute error function reweight coefficients."""
    v = np.abs(v)
    return 1 - erf((v / c) ** n)


@inf_to_zero
def cauchy(v: np.ndarray, c: float = 2.385, n: float = 2.0) -> np.ndarray:
    """Compute Cauchy M-estimator reweight coefficients."""
    v = np.abs(v)
    return np.divide(1, 1 + (v / c) ** n)


@inf_to_zero
def t(v: np.ndarray, k: int, c: float = 1.0, n: float = 2.0) -> np.ndarray:
    """Compute T distribution reweight coefficients."""
    v = np.abs(v)
    return np.power(1 + v**n / (c * k), -(k + 1) / 2)


@inf_to_zero
def chain_bell(v: np.ndarray, c: float = 1.0, n: float = 1.0) -> np.ndarray:
    """Compute chain bell function reweight coefficients."""
    v = np.abs(v)
    return np.divide(1, np.cosh(np.divide(v**n * np.e, 2 * c)))


# TRIGONOMETRIC FUNCTIONS -----------------------------
@inf_to_zero
def chain(v: np.ndarray, c: float = 1.0) -> np.ndarray:
    """Compute chain function reweight coefficients."""
    v = np.abs(v)
    coeff = -np.cosh((v * np.e) / (2 * c)) + 2
    return np.where(coeff < 0, 0, coeff)


@inf_to_zero
def andrews(v: np.ndarray, c: float = 4.207) -> np.ndarray:
    """Compute Andrews M-estimator reweight coefficients."""
    v = np.abs(v)
    formula = lambda x: np.sinc(x / c)
    return apply_where(v, v <= c, formula, np.zeros_like)


@inf_to_zero
def wave(v: np.ndarray, c: float = 2.5) -> np.ndarray:
    """Compute wave function reweight coefficients."""
    v = np.abs(v)
    formula = lambda x: np.divide(np.cos(x * np.pi / c) + 1, 2)
    return apply_where(v, v <= c, formula, np.zeros_like)


@inf_to_zero
def half_wave(v: np.ndarray, c: float = 2.5) -> np.ndarray:
    """Compute half-wave function reweight coefficients."""
    v = np.abs(v)
    formula = lambda x: np.cos(x * np.pi / (2 * c))
    return apply_where(v, v <= c, formula, np.zeros_like)


# OTHER ----------------------------------------------------
@inf_to_zero
def wigner(v: np.ndarray, c: float = 3.137) -> np.ndarray:
    """Compute Wigner M-estimator reweight coefficients."""
    v = np.abs(v)
    formula = lambda x: np.sqrt(1 - (x / c) ** 2)
    return apply_where(v, v <= c, formula, np.zeros_like)


@inf_to_zero
def ellipse_curve(v: np.ndarray, c: float = 2.5) -> np.ndarray:
    """Compute ellipse curve function reweight coefficients."""
    v = np.abs(v)
    c2 = 2 * c
    coeff = np.zeros_like(v)

    mask1 = v <= c
    mask2 = (v > c) & (v <= c2)

    coeff[mask1] = (1 + np.sqrt(1 - (v[mask1] / c) ** 2)) / 2
    coeff[mask2] = (1 - np.sqrt(1 - ((v[mask2] - c2) / c) ** 2)) / 2

    return coeff


@inf_to_zero
def trim(v: np.ndarray, c: float = 2.5) -> np.ndarray:
    """Compute trimming function reweight coefficients."""
    return apply_where(v, v <= c, np.ones_like, np.zeros_like)
