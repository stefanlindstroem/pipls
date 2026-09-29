"""Defensive array-storage helpers for immutable public results."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.intp]


def _read_only_float_array(
    value: ArrayLike,
    *,
    name: str,
    ndim: int = 1,
    require_finite: bool = True,
) -> FloatArray:
    """Return a defensive read-only float64 copy with basic storage checks."""

    array = np.array(value, dtype=np.float64, copy=True)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional; got shape {array.shape}.")
    if require_finite and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    array.setflags(write=False)
    return array


def _read_only_int_array(
    value: ArrayLike,
    *,
    name: str,
    ndim: int = 1,
) -> IntArray:
    """Return a defensive read-only platform-integer copy."""

    raw = np.asarray(value)
    if raw.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional; got shape {raw.shape}.")
    if raw.dtype.kind not in "iu" or raw.dtype.kind == "b":
        raise ValueError(f"{name} must contain integers.")
    array = np.array(raw, dtype=np.intp, copy=True)
    array.setflags(write=False)
    return array
