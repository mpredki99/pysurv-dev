# Coding: UTF-8
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
    "sigma",
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


# WITH TOLERANCE
def huber(v, c=1.345):
    v = np.abs(v)
    return np.where(v > c, c / v, 1)


def slope(v, c=2, a=2):
    v = np.abs(v)
    c, a = c
    weights = 1 + (c - v) / a
    return np.clip(weights, 0, 1)


def hampel(v, a=1.7, b=3.4, c=8.5):
    v = np.abs(v)
    a, b, c = c
    weights = np.ones_like(v)
    weights[v > a] = np.divide(a, v[v > a])
    weights[v > b] = np.divide(a, v[v > b]) * np.divide(c - v[v > b], c - b)
    weights[v > c] = 0
    return weights


def danish(v, c=2.5):
    v = np.abs(v)
    return np.where(v > c, np.exp(-v / c), 1)


# BELL CURVES
def epanechnikov(v, c=3.674, n=2.0):
    v = np.abs(v)
    return np.where(v <= c, 1 - (v / c) ** n, 0)


def tukey(v, c=4.685, n=2.0):
    v = np.abs(v)
    return np.where(v <= c, (1 - (v / c) ** n) ** n, 0)


def jacobi(v, c=4.687, n=1.0):
    v = np.abs(v)
    return np.where(v <= c, (1 - (v / c) ** n) ** n * (1 + (v / c) ** n) ** n, 0)


def exponential(v, c=2.0, n=2.0):
    v = np.abs(v)
    return np.exp(-((v / c) ** n))


def sigma(v, sigma_sq, c=2.0, n=2.0):
    v = np.abs(v)
    return np.exp(-(v ** n) / (sigma_sq * c))


def error_func(v, c=1.414, n=2.0):
    v = np.abs(v)
    return 1 - erf((v / c) ** n)


def cauchy(v, c=2.385, n=2.0):
    v = np.abs(v)
    return np.divide(1, 1 + (v / c) ** n)


def t(v, k, c=1.0, n=2.0):
    v = np.abs(v)
    return np.power(1 + v ** n / (c * k), -(k + 1) / 2)


def chain_bell(v, c=1.0, n=1.0):
    v = np.abs(v)
    return np.divide(1, np.cosh(np.divide(v ** n * np.e, 2 * c)))


# TRIGONOMETRIC FUNCTIONS
def chain(v, c=1.0):
    v = np.abs(v)
    weights = -np.cosh((v * np.e) / (2 * c)) + 2
    return np.where(weights < 0, 0, weights)


def andrews(v, c=4.207):
    v = np.abs(v)
    return np.where(v <= c, np.sinc(v / c), 0)


def wave(v, c=2.5):
    v = np.abs(v)
    return np.where(v <= c, np.divide(np.cos(v * np.pi / c) + 1, 2), 0)


def half_wave(v, c=2.5):
    v = np.abs(v)
    return np.where(v <= c, np.cos(v * np.pi / (2 * c)), 0)


# OTHERS
def wigner(v, c=3.137):
    v = np.abs(v)
    return np.where(v <= c, np.sqrt(1 - (v / c) ** 2), 0)


def ellipse_curve(v, c=2.5):
    v = np.abs(v)
    c_2 = 2 * c
    weights = np.ones_like(v)
    mask1 = v <= c
    mask2 = (v > c) & (v <= c_2)
    weights[mask1] = np.divide(1 + np.sqrt(1 - (v[mask1] / c) ** 2), 2)
    weights[mask2] = np.divide(1 - np.sqrt(1 - ((v[mask2] - c_2) / c) ** 2), 2)
    weights[v > c_2] = 0
    return weights


def trim(v, c=2.5):
    v = np.abs(v)
    return np.where(v <= c, 1, 0)
