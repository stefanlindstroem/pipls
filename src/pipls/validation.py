"""Structured out-of-fold reporting for public Π-PLS estimators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ._result_validation import _read_only_int_array
from .component_path import PiPLSSelection

__all__ = [
    "PiPLSOOFReport",
]

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.intp]


@dataclass(frozen=True)
class PiPLSOOFReport:
    """Immutable OOF diagnostics for one existing Π-PLS selection.

    Parameters
    ----------
    selection : PiPLSSelection
        Exact immutable selection evaluated by the report.
    oof_predictions : ndarray
        Ordered OOF predictions. One-dimensional responses produce shape
        ``(n_samples,)``; multi-output responses produce
        ``(n_samples, n_targets)``. Covered rows are finite; uncovered rows are
        represented entirely by NaN.
    oof_prediction_counts : ndarray of shape (n_samples,)
        Nonnegative number of validation predictions contributing to each OOF row.
    pooled_oof_r2 : float or None, default=None
        Finite pooled $R^2$ over rows with OOF coverage.

    Notes
    -----
    A report returned by :meth:`pipls.PiPLSSearchCV.oof_report` reuses the search
    splits that produced the supplied selection. It is therefore a
    selection-conditioned diagnostic rather than an unbiased post-selection
    performance estimate. Arrays are defensive, read-only copies.
    """

    selection: PiPLSSelection
    oof_predictions: FloatArray
    oof_prediction_counts: IntArray
    pooled_oof_r2: float | None = None

    def __post_init__(self) -> None:
        """Store defensive read-only copies of OOF arrays."""

        predictions = np.array(self.oof_predictions, dtype=np.float64, copy=True)
        predictions.setflags(write=False)
        object.__setattr__(self, "oof_predictions", predictions)
        object.__setattr__(
            self,
            "oof_prediction_counts",
            _read_only_int_array(
                self.oof_prediction_counts,
                name="oof_prediction_counts",
            ),
        )

    def __reduce__(self) -> tuple[type[PiPLSOOFReport], tuple[object, ...]]:
        """Reconstruct so unpickled arrays remain read-only."""

        return (
            type(self),
            (
                self.selection,
                self.oof_predictions,
                self.oof_prediction_counts,
                self.pooled_oof_r2,
            ),
        )

    @property
    def has_complete_oof_coverage(self) -> bool:
        """Whether every input row received at least one validation prediction."""

        return bool(np.all(self.oof_prediction_counts > 0))
