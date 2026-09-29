"""Evaluate a standard PLS component path for comparison with Π-PLS."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

ALGORITHM = "NIPALS"
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.intp]
CVSplit = tuple[ArrayLike, ArrayLike]
CVSpec = KFold | Sequence[CVSplit]


@dataclass(frozen=True)
class PLSComponentPath:
    """Immutable ordinary-PLS component-path results.

    Arrays contain one result per evaluated component count and are defensive,
    read-only copies aligned by row.
    """

    n_components: IntArray
    cv_mse_mean: FloatArray
    cv_mse_std: FloatArray
    algorithm: str
    n_splits: int

    def __post_init__(self) -> None:
        n_components = _read_only_int_array(self.n_components, name="n_components")
        cv_mse_mean = _read_only_float_array(self.cv_mse_mean, name="cv_mse_mean")
        cv_mse_std = _read_only_float_array(
            self.cv_mse_std,
            name="cv_mse_std",
        )
        if n_components.size == 0:
            raise ValueError("A PLS component path must contain at least one result.")
        if (
            cv_mse_mean.size != n_components.size
            or cv_mse_std.size != n_components.size
        ):
            raise ValueError("All PLS component-path arrays must have the same length.")
        if np.any(n_components <= 0):
            raise ValueError("n_components must contain positive integers.")
        if np.any(np.diff(n_components) <= 0):
            raise ValueError("n_components must be unique and strictly ascending.")
        if not np.isfinite(cv_mse_mean).all():
            raise ValueError("cv_mse_mean must contain only finite values.")
        if not np.isfinite(cv_mse_std).all() or np.any(cv_mse_std < 0.0):
            raise ValueError("cv_mse_std must contain finite nonnegative values.")
        algorithm = str(self.algorithm).strip()
        if not algorithm:
            raise ValueError("algorithm must be a nonempty string.")
        if isinstance(self.n_splits, bool) or not isinstance(self.n_splits, (int, np.integer)):
            raise ValueError("n_splits must be a positive integer.")
        n_splits = int(self.n_splits)
        if n_splits <= 0:
            raise ValueError("n_splits must be a positive integer.")

        object.__setattr__(self, "n_components", n_components)
        object.__setattr__(self, "cv_mse_mean", cv_mse_mean)
        object.__setattr__(self, "cv_mse_std", cv_mse_std)
        object.__setattr__(self, "algorithm", algorithm)
        object.__setattr__(self, "n_splits", n_splits)

    def __reduce__(self) -> tuple[type[PLSComponentPath], tuple[object, ...]]:
        """Reconstruct through validation so unpickled arrays remain read-only."""

        return (
            type(self),
            (
                self.n_components,
                self.cv_mse_mean,
                self.cv_mse_std,
                self.algorithm,
                self.n_splits,
            ),
        )


def evaluate_pls_component_path(
    X: ArrayLike,
    Y: ArrayLike,
    *,
    max_n_components: int,
    cv: CVSpec,
) -> PLSComponentPath:
    """Return fold-local response-standardized CV-MSE for standard PLS.

    One maximum-component ``PLSRegression`` fit is evaluated per fold. NIPALS
    extracts components sequentially, so truncating the fitted rotations and
    response loadings gives the same nested path without repeated earlier fits.
    ``cv`` may be a ``KFold`` splitter or a reusable materialized sequence of
    ``(train, validation)`` index pairs.
    """

    X_array = np.asarray(X, dtype=np.float64)
    Y_array = np.asarray(Y, dtype=np.float64)
    if X_array.ndim != 2 or Y_array.ndim != 2:
        raise ValueError("X and Y must be two-dimensional matrices.")
    if X_array.shape[0] != Y_array.shape[0]:
        raise ValueError(
            "Predictor and response row counts differ: "
            f"{X_array.shape[0]} != {Y_array.shape[0]}."
        )
    if not np.isfinite(X_array).all() or not np.isfinite(Y_array).all():
        raise ValueError("X and Y must contain only finite values.")
    if max_n_components < 1:
        raise ValueError("max_n_components must be at least 1.")
    algebraic_max = min(Y_array.shape[1], X_array.shape[1], X_array.shape[0] - 1)
    if max_n_components > algebraic_max:
        raise ValueError(
            "max_n_components exceeds the centered-data algebraic limit: "
            f"{max_n_components} > {algebraic_max}."
        )

    splits = _materialize_cv_splits(cv, X_array, Y_array)
    n_splits = len(splits)
    split_mse = np.empty((max_n_components, n_splits), dtype=np.float64)
    for split_index, (train, validation) in enumerate(splits):
        X_train = X_array[train]
        Y_train = Y_array[train]
        X_validation = X_array[validation]
        Y_validation = Y_array[validation]

        x_mean = np.mean(X_train, axis=0)
        y_mean = np.mean(Y_train, axis=0)
        x_scale = np.std(X_train, axis=0, ddof=1)
        y_scale = np.std(Y_train, axis=0, ddof=1)
        x_scale = np.where(x_scale == 0.0, 1.0, x_scale)
        y_scale = np.where(y_scale == 0.0, 1.0, y_scale)

        X_train_scaled = (X_train - x_mean) / x_scale
        Y_train_scaled = (Y_train - y_mean) / y_scale
        X_validation_scaled = (X_validation - x_mean) / x_scale
        Y_validation_scaled = (Y_validation - y_mean) / y_scale

        model = PLSRegression(n_components=max_n_components, scale=False)
        model.fit(X_train_scaled, Y_train_scaled)
        for n_components in range(1, max_n_components + 1):
            x_weights = model.x_weights_[:, :n_components]
            x_loadings = model.x_loadings_[:, :n_components]
            y_loadings = model.y_loadings_[:, :n_components]
            x_rotations = x_weights @ np.linalg.pinv(x_loadings.T @ x_weights)
            prediction_scaled = X_validation_scaled @ x_rotations @ y_loadings.T
            split_mse[n_components - 1, split_index] = float(
                np.mean((Y_validation_scaled - prediction_scaled) ** 2)
            )

    return PLSComponentPath(
        n_components=np.arange(1, max_n_components + 1, dtype=np.intp),
        cv_mse_mean=np.mean(split_mse, axis=1),
        cv_mse_std=np.std(split_mse, axis=1),
        algorithm=ALGORITHM,
        n_splits=n_splits,
    )


def _materialize_cv_splits(
    cv: CVSpec,
    X: FloatArray,
    Y: FloatArray,
) -> tuple[tuple[IntArray, IntArray], ...]:
    if isinstance(cv, KFold):
        raw_splits = cv.split(X, Y)
    else:
        raw_splits = iter(cv)

    splits: list[tuple[IntArray, IntArray]] = []
    n_samples = X.shape[0]
    for split_index, (train, validation) in enumerate(raw_splits):
        train_array = np.asarray(train, dtype=np.intp)
        validation_array = np.asarray(validation, dtype=np.intp)
        if train_array.ndim != 1 or validation_array.ndim != 1:
            raise ValueError(
                f"CV split {split_index} indices must be one-dimensional."
            )
        if train_array.size == 0 or validation_array.size == 0:
            raise ValueError(f"CV split {split_index} must have nonempty partitions.")
        if (
            np.any(train_array < 0)
            or np.any(train_array >= n_samples)
            or np.any(validation_array < 0)
            or np.any(validation_array >= n_samples)
        ):
            raise ValueError(f"CV split {split_index} contains an out-of-range index.")
        splits.append((train_array, validation_array))

    if not splits:
        raise ValueError("cv must provide at least one validation split.")
    return tuple(splits)


def _read_only_int_array(value: ArrayLike, *, name: str) -> IntArray:
    array = np.array(value, dtype=np.intp, copy=True)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {array.shape}.")
    array.setflags(write=False)
    return array


def _read_only_float_array(value: ArrayLike, *, name: str) -> FloatArray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {array.shape}.")
    array.setflags(write=False)
    return array
