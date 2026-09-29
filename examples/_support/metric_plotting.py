"""Example-local helpers for metric-plot presentation contracts."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def response_r2_ylim(
    values: ArrayLike,
    *,
    negative_padding_fraction: float = 0.05,
) -> tuple[float, float]:
    """Return maintained y limits for a response-wise R² bar plot.

    The upper limit is exactly 1.0. The lower limit is 0.0 when all displayed
    values are nonnegative and extends below the most negative value otherwise.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("values must contain at least one R² value.")
    if not np.isfinite(array).all():
        raise ValueError("values must contain only finite R² values.")
    if (
        not np.isfinite(negative_padding_fraction)
        or negative_padding_fraction < 0.0
    ):
        raise ValueError("negative_padding_fraction must be finite and nonnegative.")

    minimum = float(np.min(array))
    if minimum >= 0.0:
        return 0.0, 1.0

    padding = negative_padding_fraction * (1.0 - minimum)
    return minimum - padding, 1.0
