"""Scoring utilities for Π-PLS model selection."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.utils.validation import check_is_fitted

FloatArray = NDArray[np.float64]


__all__ = [
    "neg_response_standardized_mse",
    "response_standardized_mse",
]


def response_standardized_mse(
    estimator: Any,
    X: ArrayLike,
    y: ArrayLike,
) -> float:
    r"""Return response-standardized mean squared error.

    For response $j$, residuals are divided by the sample standard deviation
    estimated from the estimator's training responses. The returned scalar is the
    uniform mean over observations and responses. The scaling is independent of
    whether the estimator itself was fitted with response scaling.

    Parameters
    ----------
    estimator : estimator
        Fitted Π-PLS estimator, or pipeline ending in one. The scorer uses
        private response-scale state learned during fitting.
    X : array-like of shape (n_samples, n_features)
        Predictor observations to score.
    y : array-like of shape (n_samples,) or (n_samples, n_targets)
        Observed responses.

    Returns
    -------
    float
        Nonnegative response-standardized MSE.
    """

    response_scale = _response_scale_for_scoring(estimator)
    return _response_standardized_mse(
        y,
        estimator.predict(X),
        response_scale,
    )


def neg_response_standardized_mse(
    estimator: Any,
    X: ArrayLike,
    y: ArrayLike,
) -> float:
    r"""Return negative response-standardized MSE.

    This sign-reversed form follows the scikit-learn scorer convention that larger
    scores are better. Maximizing it is equivalent to minimizing
    :func:`response_standardized_mse`.

    Parameters
    ----------
    estimator : estimator
        Fitted Π-PLS estimator or compatible pipeline.
    X : array-like of shape (n_samples, n_features)
        Predictor observations to score.
    y : array-like of shape (n_samples,) or (n_samples, n_targets)
        Observed responses.

    Returns
    -------
    float
        Nonpositive negative response-standardized MSE.
    """

    return -response_standardized_mse(estimator, X, y)


def _training_response_scale(y_train: ArrayLike) -> FloatArray:
    """Return training response scales using ``ddof=1`` and unit zero scales."""

    y_array = _as_2d_targets(y_train, name="y_train")
    centered = y_array - _safe_column_mean(y_array)
    if not np.all(np.isfinite(centered)):
        raise ValueError("Centering y_train produced nonfinite values.")
    return _safe_sample_scale(centered)


def _safe_column_mean(values: FloatArray) -> FloatArray:
    """Return ordinary column means, with a range-safe overflow fallback."""

    with np.errstate(over="ignore", invalid="ignore"):
        mean = np.mean(values, axis=0)
    failed = ~np.isfinite(mean)
    if np.any(failed):
        magnitude = np.max(np.abs(values[:, failed]), axis=0)
        mean[failed] = (
            np.mean(values[:, failed] / magnitude, axis=0) * magnitude
        )
    return np.asarray(mean, dtype=np.float64)


def _safe_sample_scale(centered: FloatArray) -> FloatArray:
    """Return ordinary sample scales, with range-safe boundary fallbacks."""

    if centered.shape[0] <= 1:
        return np.ones(centered.shape[1], dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        scale = np.std(centered, axis=0, ddof=1)
    magnitude = np.max(np.abs(centered), axis=0)
    failed = (~np.isfinite(scale)) | ((scale == 0.0) & (magnitude > 0.0))
    if np.any(failed):
        normalized_scale = np.std(
            centered[:, failed] / magnitude[failed],
            axis=0,
            ddof=1,
        )
        computed = normalized_scale * magnitude[failed]
        scale[failed] = np.where(computed > 0.0, computed, magnitude[failed])
    return np.where(scale == 0.0, 1.0, scale).astype(np.float64, copy=False)


def _response_standardized_mse(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    response_scale: ArrayLike,
) -> float:
    """Return uniformly response-weighted MSE after response scaling."""

    true_array = _as_2d_targets(y_true, name="y_true")
    pred_array = _as_2d_targets(y_pred, name="y_pred")
    if true_array.shape != pred_array.shape:
        raise ValueError(
            "y_true and y_pred must have identical shapes: "
            f"got {true_array.shape} and {pred_array.shape}."
        )
    scale = np.asarray(response_scale, dtype=np.float64)
    if scale.ndim != 1 or scale.shape[0] != true_array.shape[1]:
        raise ValueError(
            "response_scale must contain one value per response: "
            f"expected {(true_array.shape[1],)}, got {scale.shape}."
        )
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("response_scale must contain positive finite values.")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        standardized_residual = (true_array - pred_array) / scale[None, :]
        mse = float(np.mean(np.square(standardized_residual)))
    if not np.isfinite(mse):
        raise ValueError(
            "Response-standardized MSE is not representable as a finite float64 value."
        )
    return mse


def _response_scale_for_scoring(estimator: Any) -> ArrayLike:
    """Return the fitted response scale from a direct estimator or final pipeline step."""

    if hasattr(estimator, "_response_scale_for_scoring_"):
        check_is_fitted(estimator, attributes=["_response_scale_for_scoring_"])
        return cast(ArrayLike, estimator._response_scale_for_scoring_)
    steps = getattr(estimator, "steps", None)
    if steps:
        final_estimator = steps[-1][1]
        check_is_fitted(final_estimator, attributes=["_response_scale_for_scoring_"])
        return cast(ArrayLike, final_estimator._response_scale_for_scoring_)
    raise ValueError("The estimator does not contain fitted Pi-PLS response scaling state.")


def _as_2d_targets(values: ArrayLike, *, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(f"{name} must be one- or two-dimensional; got shape {array.shape}.")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one sample and one response.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array
