"""Small compatibility helpers across supported scikit-learn releases."""

from __future__ import annotations

from typing import Any, cast

from sklearn.base import BaseEstimator

try:
    from sklearn.utils.validation import validate_data as _public_validate_data
except ImportError:  # pragma: no cover - exercised with scikit-learn 1.4/1.5
    _public_validate_data = None


def _validate_estimator_data(
    estimator: BaseEstimator,
    X: Any,
    y: Any = "no_validation",
    *,
    reset: bool,
    **kwargs: Any,
) -> Any:
    """Use public ``validate_data`` when available, else the legacy method."""

    if _public_validate_data is not None:
        return _public_validate_data(estimator, X, y, reset=reset, **kwargs)
    legacy = cast(Any, estimator._validate_data)
    return legacy(X, y, reset=reset, **kwargs)
