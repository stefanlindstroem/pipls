"""Dataset containers and validation support."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PiPLSDataset:
    r"""Validated multivariate regression dataset.

    Plain arrays and data frames passed directly to ``fit(X, Y)`` remain the
    primary real-data interface. This container carries packaged reference
    datasets and structured experiment data.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Numeric predictor matrix.
    Y : array-like of shape (n_samples,) or (n_samples, n_targets)
        Numeric response data. One-dimensional input is stored as one column.
    feature_names : sequence of str
        Unique nonempty predictor names.
    target_names : sequence of str
        Unique nonempty response names.
    metadata : mapping of str to object, default={}
        Dataset metadata. The top-level mapping is shallow-copied.
    Attributes
    ----------
    X, Y : ndarray
        Read-only ``float64`` predictor and two-dimensional response matrices.
    feature_names, target_names : tuple of str
        Validated axis labels.
    metadata : mapping
        Shallow-copied metadata mapping.
    """

    X: FloatArray
    Y: FloatArray
    feature_names: Sequence[str]
    target_names: Sequence[str]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        X = _validated_matrix(self.X, name="X", allow_vector=False)
        Y = _validated_matrix(self.Y, name="Y", allow_vector=True)
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must contain the same number of samples.")
        if X.shape[0] == 0 or X.shape[1] == 0 or Y.shape[1] == 0:
            raise ValueError("X and Y must contain at least one sample and one column.")

        feature_names = _validated_names(
            self.feature_names,
            expected=X.shape[1],
            name="feature_names",
        )
        target_names = _validated_names(
            self.target_names,
            expected=Y.shape[1],
            name="target_names",
        )
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping.")
        metadata = dict(self.metadata)

        object.__setattr__(self, "X", X)
        object.__setattr__(self, "Y", Y)
        object.__setattr__(self, "feature_names", feature_names)
        object.__setattr__(self, "target_names", target_names)
        object.__setattr__(self, "metadata", metadata)

    @property
    def n_samples(self) -> int:
        """Number of aligned observations."""

        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        """Number of predictor columns."""

        return int(self.X.shape[1])

    @property
    def n_targets(self) -> int:
        """Number of response columns."""

        return int(self.Y.shape[1])


def _validated_matrix(
    value: object,
    *,
    name: str,
    allow_vector: bool,
) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numeric values.")
    if allow_vector and raw.ndim == 1:
        raw = raw.reshape(-1, 1)
    if raw.ndim != 2:
        expected = "one or two dimensions" if allow_vector else "two dimensions"
        raise ValueError(f"{name} must have {expected}.")
    array = np.array(raw, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    array.setflags(write=False)
    return array


def _validated_names(
    values: Sequence[str],
    *,
    expected: int,
    name: str,
) -> tuple[str, ...]:
    result = tuple(values)
    if len(result) != expected:
        raise ValueError(f"{name} must contain exactly {expected} values.")
    if any(not isinstance(value, str) or not value.strip() for value in result):
        raise TypeError(f"{name} must contain non-empty strings.")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique values.")
    return result
