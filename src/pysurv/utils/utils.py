# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

from functools import wraps
from typing import Callable

import numpy as np


def inf_to_zero(func):
    """Decorator that turns infinite values to 0 (robust methods have limit 0 with v -> inf)."""

    @wraps(func)
    def wrapper(v, *args, **kwargs):
        is_finite_mask = np.isfinite(v)
        coeff = np.nan_to_num(v, nan=np.nan, neginf=0, posinf=0)
        coeff[is_finite_mask] = func(v[is_finite_mask], *args, **kwargs)
        return coeff

    return wrapper


def apply_where(
    arg: np.ndarray,
    cond: np.ndarray,
    func_true: Callable[[np.ndarray], np.ndarray],
    func_false: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """
    Apply `func_true` to elements of `arg` where `cond` is True,
    and `func_false` to elements where `cond` is False.

    Returns a new array with the same shape as `arg`.
    """
    result = np.empty(arg.shape)

    if np.any(cond):
        result[cond] = func_true(arg[cond])
    if np.any(~cond):
        result[~cond] = func_false(arg[~cond])

    return result
