"""Deterministic synthetic data generator."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
NoiseSpec: TypeAlias = float | tuple[float, float]

_UINT32_MAX = 2**32 - 1


def make_synthetic_data(
    *,
    n_samples: int,
    n_features: int,
    n_targets: int,
    n_shared: int,
    n_predictor_specific: int = 0,
    n_response_specific: int = 0,
    noise: NoiseSpec = 0.0,
    random_state: int = 0,
) -> tuple[FloatArray, FloatArray]:
    r"""Generate the Gaussian latent geometry used in the companion publication.

    The function implements the latent-geometry equation in the Synthetic
    generator section of the dataset API reference. Every entry of the three
    latent-score matrices and four loading matrices is drawn independently
    from $\mathcal{N}(0,1)$. Predictor and response
    noise entries are independent Gaussian draws with the requested standard
    deviations. No score centering, score standardization, loading
    orthonormalization, latent-strength scaling, or observed-variable scaling is
    applied.

    Parameters
    ----------
    n_samples : int
        Number of observations; at least one.
    n_features : int
        Number of predictor variables $p$.
    n_targets : int
        Number of response variables $q$.
    n_shared : int
        Shared latent dimension $d_{\mathrm{s}}$.
    n_predictor_specific : int, default=0
        Predictor-specific latent dimension $d_{\mathrm{p}}$.
    n_response_specific : int, default=0
        Response-specific latent dimension $d_{\mathrm{r}}$.
    noise : float or tuple of float, default=0.0
        Common noise standard deviation, or separate
        ``(sigma_x, sigma_y)`` values.
    random_state : int, default=0
        Local deterministic unsigned 32-bit random seed.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Generated predictor matrix.
    Y : ndarray of shape (n_samples, n_targets)
        Generated response matrix.

    Notes
    -----
    The generator implements the synthetic data model described by Agrawal,
    Nilsson, and Lindström (2026), *Panoramic Partial Least Squares (Π-PLS): A
    transparent and parsimonious multivariate regression model with paired
    latent directions*, *Computers & Chemical Engineering*, article 109913,
    https://doi.org/10.1016/j.compchemeng.2026.109913.
    """

    n_samples = _positive_integer(n_samples, name="n_samples")
    n_features = _positive_integer(n_features, name="n_features")
    n_targets = _positive_integer(n_targets, name="n_targets")
    n_shared = _nonnegative_integer(n_shared, name="n_shared")
    n_predictor_specific = _nonnegative_integer(
        n_predictor_specific,
        name="n_predictor_specific",
    )
    n_response_specific = _nonnegative_integer(
        n_response_specific,
        name="n_response_specific",
    )
    if n_predictor_specific + n_shared > n_features:
        raise ValueError("n_predictor_specific + n_shared must not exceed n_features.")
    if n_shared + n_response_specific > n_targets:
        raise ValueError("n_shared + n_response_specific must not exceed n_targets.")

    sigma_x, sigma_y = _resolve_noise(noise)
    seed = _validated_seed(random_state)
    rng = np.random.default_rng(seed)

    predictor_specific_scores = rng.standard_normal(
        (n_samples, n_predictor_specific)
    )
    shared_scores = rng.standard_normal((n_samples, n_shared))
    response_specific_scores = rng.standard_normal(
        (n_samples, n_response_specific)
    )
    predictor_specific_loadings = rng.standard_normal(
        (n_predictor_specific, n_features)
    )
    shared_predictor_loadings = rng.standard_normal((n_shared, n_features))
    shared_response_loadings = rng.standard_normal((n_shared, n_targets))
    response_specific_loadings = rng.standard_normal(
        (n_response_specific, n_targets)
    )
    x_noise = sigma_x * rng.standard_normal((n_samples, n_features))
    y_noise = sigma_y * rng.standard_normal((n_samples, n_targets))

    x_signal = (
        predictor_specific_scores @ predictor_specific_loadings
        + shared_scores @ shared_predictor_loadings
    )
    y_signal = (
        shared_scores @ shared_response_loadings
        + response_specific_scores @ response_specific_loadings
    )
    X = x_signal + x_noise
    Y = y_signal + y_noise
    return X, Y


def _positive_integer(value: object, *, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return result


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return result


def _resolve_noise(value: NoiseSpec) -> tuple[float, float]:
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError("noise must be a scalar or a two-tuple (x_noise, y_noise).")
        x_noise, y_noise = (float(value[0]), float(value[1]))
    else:
        x_noise = y_noise = float(value)
    if not np.isfinite(x_noise) or not np.isfinite(y_noise):
        raise ValueError("noise values must be finite.")
    if x_noise < 0.0 or y_noise < 0.0:
        raise ValueError("noise values must be nonnegative.")
    return x_noise, y_noise


def _validated_seed(value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError("random_state must be an integer.")
    seed = int(value)
    if seed < 0 or seed > _UINT32_MAX:
        raise ValueError(f"random_state must lie in [0, {_UINT32_MAX}].")
    return seed
