"""Public fixed-parameter Π-PLS regression estimator."""

from __future__ import annotations

import warnings
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    MultiOutputMixin,
    RegressorMixin,
    TransformerMixin,
)
from sklearn.metrics import r2_score
from sklearn.utils.validation import check_array, check_is_fitted

from ._core import (
    ResponseSubspace,
    SVDSolver,
    _as_positive_int,
    _decompose_predictors,
    _validate_random_state,
    _validate_response_subspace,
    fit_pipls_core,
)
from ._sklearn_compat import _validate_estimator_data
from .decomposition import PiPLSDecomposition
from .exceptions import PredictorRankSupportWarning
from .metrics import _safe_column_mean, _safe_sample_scale, _training_response_scale

__all__ = ["PiPLSRegression"]

FloatArray = NDArray[np.float64]
_MIN_TRUSTED_SAMPLES_PER_PREDICTOR_RANK = 3.0


class PiPLSRegression(
    ClassNamePrefixFeaturesOutMixin,  # type: ignore[misc]
    TransformerMixin,  # type: ignore[misc]
    RegressorMixin,  # type: ignore[misc]
    MultiOutputMixin,  # type: ignore[misc]
    BaseEstimator,  # type: ignore[misc]
):
    r"""Π-PLS regression for one fixed pair $(h, r_\pi)$.

    Parameters
    ----------
    n_components : int
        Required number of paired latent modes $h$. It must satisfy
        ``1 <= n_components <= min(n_targets, predictor_rank)``.
    predictor_rank : int
        Required retained predictor-subspace dimension $r_\pi$. After
        centering, it
        must not exceed ``min(n_features, n_samples - 1)``.
    response_subspace : {"cross_covariance", "least_squares"}, default="cross_covariance"
        Response-subspace construction used to obtain the intermediate response
        basis $\mathbf{C}$. ``"cross_covariance"`` is the peer-reviewed Π-PLS
        construction and remains the package default. ``"least_squares"`` is a
        least-squares/RRR-inspired software extension that is not part of the
        peer-reviewed companion publication.
    scale : bool, default=True
        Backward-compatible default for predictor and response scaling. If true,
        centered predictor and response columns are divided by their
        training-sample standard deviations. If false, both blocks remain only
        centered. ``scale_x`` and ``scale_y`` override this value independently
        when they are not ``None``.
    scale_x : bool or None, default=None
        Predictor-scaling override. ``None`` inherits ``scale``. If true,
        centered predictor columns are divided by their training-sample standard
        deviations; if false, predictors remain only centered.
    scale_y : bool or None, default=None
        Response-scaling override. ``None`` inherits ``scale``. If true,
        centered response columns are divided by their training-sample standard
        deviations; if false, responses remain only centered.
    copy : bool, default=True
        Whether fitting may copy the supplied arrays before preprocessing.
    svd_solver : {"auto", "full", "randomized"}, default="auto"
        Predictor SVD policy. ``"full"`` uses the exact thin SVD,
        ``"randomized"`` always uses randomized truncated SVD, and ``"auto"``
        uses randomized SVD only for sufficiently large matrices and low retained
        rank. The response-side and coupling SVDs always remain exact.
    random_state : int, numpy.random.RandomState or None, default=0
        Random state used by randomized predictor SVD. The default integer seed
        makes ``"auto"`` and ``"randomized"`` reproducible. ``None`` follows
        NumPy's global random state.

    Attributes
    ----------
    n_features_in_ : int
        Number of predictor columns seen during fitting.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Predictor names seen during fitting. Defined only when all input feature
        names are strings.
    n_targets_ : int
        Number of response columns seen during fitting.
    max_predictor_rank_ : int
        Algebraic upper bound ``min(n_features_in_, n_samples - 1)`` for the
        fitted data.
    decomposition_ : PiPLSDecomposition
        Immutable Π-PLS predictor directions, dilations, response directions,
        and numerical diagnostics.
    coef_ : ndarray of shape (n_targets_, n_features_in_)
        Regression coefficients in original predictor and response units.
    intercept_ : ndarray of shape (n_targets_,)
        Regression intercept in original response units.
    x_mean_ : ndarray of shape (n_features_in_,)
        Predictor means learned from the training data.
    y_mean_ : ndarray of shape (n_targets_,)
        Response means learned from the training data.
    x_scale_ : ndarray of shape (n_features_in_,)
        Predictor scales learned from the training data, or ones when effective
        predictor scaling is disabled. Constant columns receive scale one.
    y_scale_ : ndarray of shape (n_targets_,)
        Response scales learned from the training data, or ones when effective
        response scaling is disabled. Constant columns receive scale one.
    x_rotations_ : ndarray of shape (n_features_in_, n_components)
        Predictor directions.
    y_rotations_ : ndarray of shape (n_targets_, n_components)
        Response directions.
    x_scores_ : ndarray of shape (n_samples, n_components)
        Training predictor scores.
    y_scores_ : ndarray of shape (n_samples, n_components)
        Training response scores.
    x_loadings_ : ndarray of shape (n_features_in_, n_components)
        Least-squares predictor loadings for reconstructing centered/scaled X.
    y_loadings_ : ndarray of shape (n_targets_, n_components)
        Least-squares response loadings for reconstructing centered/scaled y.

    Notes
    -----
    This estimator performs no parameter selection or cross-validation. Both
    rank parameters are therefore required explicitly. Use
    :class:`pipls.PiPLSSearchCV` when paired-mode count and predictor rank must be
    selected by cross-validation.
    """

    def __init__(
        self,
        *,
        n_components: int,
        predictor_rank: int,
        response_subspace: ResponseSubspace = "cross_covariance",
        scale: bool = True,
        scale_x: bool | None = None,
        scale_y: bool | None = None,
        copy: bool = True,
        svd_solver: SVDSolver = "auto",
        random_state: int | np.random.RandomState | None = 0,
    ) -> None:
        self.n_components = n_components
        self.scale = scale
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.copy = copy
        self.predictor_rank = predictor_rank
        self.response_subspace = response_subspace
        self.svd_solver = svd_solver
        self.random_state = random_state

    def fit(self, X: ArrayLike, y: ArrayLike) -> PiPLSRegression:
        """Fit one fixed Π-PLS model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Predictor matrix. Sparse input is not supported.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Response vector or matrix.

        Returns
        -------
        self : PiPLSRegression
            Fitted estimator.
        """

        _clear_fitted_state(self)
        try:
            return self._fit(X, y)
        except np.linalg.LinAlgError as error:
            _clear_fitted_state(self)
            raise ValueError(
                "Pi-PLS fitting failed during numerical decomposition."
            ) from error
        except Exception:
            _clear_fitted_state(self)
            raise

    def _fit(self, X: ArrayLike, y: ArrayLike) -> PiPLSRegression:
        self._validate_constructor_parameters()
        validated = _validate_estimator_data(
            self,
            X,
            y,
            reset=True,
            accept_sparse=False,
            dtype=np.float64,
            multi_output=True,
            y_numeric=True,
            ensure_min_samples=2,
            copy=self.copy,
        )
        X_checked, y_checked = cast(tuple[Any, Any], validated)
        X_array = np.asarray(X_checked, dtype=np.float64)
        y_array_raw = np.array(y_checked, dtype=np.float64, copy=self.copy)
        if not X_array.flags.writeable:
            X_array = X_array.copy()
        if not y_array_raw.flags.writeable or np.may_share_memory(X_array, y_array_raw):
            y_array_raw = y_array_raw.copy()
        self._y_was_1d = y_array_raw.ndim == 1
        y_array = y_array_raw.reshape(-1, 1) if self._y_was_1d else y_array_raw

        self.n_targets_ = int(y_array.shape[1])
        if self.n_components > self.n_targets_:
            raise ValueError(
                "n_components must not exceed the number of response columns: "
                f"got n_components={self.n_components}, n_targets={self.n_targets_}."
            )

        n_samples, n_features = X_array.shape
        max_predictor_rank = min(n_features, n_samples - 1)
        predictor_rank = int(self.predictor_rank)
        if predictor_rank > max_predictor_rank:
            raise ValueError(
                "predictor_rank must satisfy predictor_rank <= min(n_features, "
                "n_samples - 1) after centering; "
                f"got predictor_rank={predictor_rank}, n_features={n_features}, "
                f"n_samples={n_samples}."
            )
        if n_samples < _MIN_TRUSTED_SAMPLES_PER_PREDICTOR_RANK * predictor_rank:
            warnings.warn(
                f"The fitted model has {n_samples / predictor_rank:.3g} samples per "
                "retained predictor-rank direction, below the recommended minimum of 3. "
                "The model may have poor statistical support; use external validation.",
                PredictorRankSupportWarning,
                stacklevel=2,
            )
        self._fit_fixed_rank(
            X_array,
            y_array,
            predictor_rank=predictor_rank,
            max_predictor_rank=max_predictor_rank,
        )
        return self

    def predict(self, X: ArrayLike, copy: bool = True) -> FloatArray:
        """Predict responses in their original units.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Predictor matrix.
        copy : bool, default=True
            Whether validation may copy ``X``.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Predictions in original response units. A one-dimensional response
            supplied to ``fit`` produces one-dimensional predictions.
        """

        check_is_fitted(self, attributes=["coef_", "intercept_"])
        X_checked = cast(
            FloatArray,
            _validate_estimator_data(
                self,
                X,
                reset=False,
                accept_sparse=False,
                dtype=np.float64,
                copy=copy,
            ),
        )
        with np.errstate(over="ignore", invalid="ignore"):
            prediction: FloatArray = (
                np.asarray(X_checked, dtype=np.float64) @ self.coef_.T + self.intercept_
            )
        _require_finite_output(prediction, operation="Prediction")
        if self._y_was_1d:
            return prediction[:, 0]
        return prediction

    def transform(
        self,
        X: ArrayLike,
        y: ArrayLike | None = None,
        copy: bool = True,
    ) -> FloatArray | tuple[FloatArray, FloatArray]:
        """Transform predictors, and optionally responses, to latent scores.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Predictor matrix.
        y : array-like of shape (n_samples,) or (n_samples, n_targets), optional
            Response vector or matrix.
        copy : bool, default=True
            Whether validation may copy the supplied arrays.

        Returns
        -------
        x_scores : ndarray of shape (n_samples, n_components)
            Predictor scores when ``y`` is omitted.
        (x_scores, y_scores) : tuple of ndarray
            Predictor and response scores when ``y`` is supplied. Both arrays
            have shape ``(n_samples, n_components)``.
        """

        check_is_fitted(self, attributes=["x_rotations_", "y_rotations_", "x_mean_", "y_mean_"])
        X_checked = cast(
            FloatArray,
            _validate_estimator_data(
                self,
                X,
                reset=False,
                accept_sparse=False,
                dtype=np.float64,
                copy=copy,
            ),
        )
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            X_cs = (np.asarray(X_checked, dtype=np.float64) - self.x_mean_) / self.x_scale_
            x_scores = X_cs @ self.x_rotations_
        _require_finite_output(x_scores, operation="Predictor transformation")
        if y is None:
            return x_scores

        y_checked = check_array(
            y,
            ensure_2d=False,
            dtype=np.float64,
            ensure_min_samples=X_cs.shape[0],
            copy=copy,
        )
        y_array = np.asarray(y_checked, dtype=np.float64)
        if y_array.ndim == 1:
            y_array = y_array.reshape(-1, 1)
        if y_array.shape[0] != X_cs.shape[0]:
            raise ValueError(
                "X and y must contain the same number of samples: "
                f"got {X_cs.shape[0]} and {y_array.shape[0]}."
            )
        if y_array.shape[1] != self.n_targets_:
            raise ValueError(
                "y has an incompatible number of targets: "
                f"expected {self.n_targets_}, got {y_array.shape[1]}."
            )
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            y_cs = (y_array - self.y_mean_) / self.y_scale_
            y_scores = y_cs @ self.y_rotations_
        _require_finite_output(y_scores, operation="Response transformation")
        return x_scores, y_scores

    def fit_transform(
        self,
        X: ArrayLike,
        y: ArrayLike | None = None,
    ) -> tuple[FloatArray, FloatArray]:
        """Fit the model and return predictor and response scores.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Predictor matrix.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Response vector or matrix. ``y`` is required.

        Returns
        -------
        x_scores, y_scores : tuple of ndarray
            Training predictor and response scores, each with shape
            ``(n_samples, n_components)``.
        """

        if y is None:
            raise ValueError("y is required to fit PiPLSRegression.")
        self.fit(X, y)
        return self.x_scores_.copy(), self.y_scores_.copy()

    def inverse_transform(
        self,
        X: ArrayLike,
        y: ArrayLike | None = None,
    ) -> FloatArray | tuple[FloatArray, FloatArray]:
        """Reconstruct predictors, and optionally responses, from latent scores.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_components)
            Predictor scores.
        y : array-like of shape (n_samples, n_components), optional
            Response scores.

        Returns
        -------
        X_reconstructed : ndarray of shape (n_samples, n_features_in_)
            Predictor reconstruction when ``y`` is omitted.
        (X_reconstructed, y_reconstructed) : tuple of ndarray
            Predictor and response reconstructions when ``y`` is supplied.

        Notes
        -----
        Reconstruction is least-squares and is exact only when the retained
        latent spaces span the corresponding centered/scaled data spaces.
        """

        check_is_fitted(self, attributes=["x_loadings_", "y_loadings_"])
        x_scores = check_array(X, ensure_2d=True, dtype=np.float64)
        if x_scores.shape[1] != self.n_components:
            raise ValueError(
                "X has an incompatible number of latent components: "
                f"expected {self.n_components}, got {x_scores.shape[1]}."
            )
        with np.errstate(over="ignore", invalid="ignore"):
            X_original = (x_scores @ self.x_loadings_.T) * self.x_scale_ + self.x_mean_
        _require_finite_output(X_original, operation="Predictor reconstruction")
        if y is None:
            return np.asarray(X_original, dtype=np.float64)

        y_scores = check_array(y, ensure_2d=False, dtype=np.float64)
        if y_scores.ndim == 1:
            y_scores = y_scores.reshape(-1, 1)
        if y_scores.shape[0] != x_scores.shape[0]:
            raise ValueError(
                "X and y scores must contain the same number of samples: "
                f"got {x_scores.shape[0]} and {y_scores.shape[0]}."
            )
        if y_scores.shape[1] != self.n_components:
            raise ValueError(
                "y has an incompatible number of latent components: "
                f"expected {self.n_components}, got {y_scores.shape[1]}."
            )
        with np.errstate(over="ignore", invalid="ignore"):
            y_original = (y_scores @ self.y_loadings_.T) * self.y_scale_ + self.y_mean_
        _require_finite_output(y_original, operation="Response reconstruction")
        return (
            np.asarray(X_original, dtype=np.float64),
            np.asarray(y_original, dtype=np.float64),
        )

    def score(self, X: ArrayLike, y: ArrayLike, sample_weight: ArrayLike | None = None) -> float:
        r"""Return uniformly averaged $R^2$ in original response units.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Predictor matrix.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Observed responses.
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights passed to :func:`sklearn.metrics.r2_score`.

        Returns
        -------
        score : float
            Uniform average of response-wise $R^2$ values.
        """

        return float(
            r2_score(
                y,
                self.predict(X),
                sample_weight=sample_weight,
                multioutput="uniform_average",
            )
        )

    def _more_tags(self) -> dict[str, bool]:
        """Legacy scikit-learn tags for releases before the Tags dataclasses."""

        return {"multioutput": True, "poor_score": True}

    def __sklearn_tags__(self) -> Any:
        """Declare multi-output regression and PLS-like score expectations."""

        parent = getattr(super(), "__sklearn_tags__", None)
        if parent is None:  # pragma: no cover - scikit-learn 1.4/1.5
            return self._more_tags()
        tags = parent()
        tags.target_tags.multi_output = True
        tags.target_tags.single_output = True
        if tags.regressor_tags is not None:
            tags.regressor_tags.poor_score = True
        return tags

    def _fit_fixed_rank(
        self,
        X: FloatArray,
        y: FloatArray,
        *,
        predictor_rank: int,
        max_predictor_rank: int,
    ) -> None:
        scale_x = self.scale if self.scale_x is None else self.scale_x
        self.x_mean_, self.x_scale_ = _preprocess_predictor_block(
            X,
            scale=bool(scale_x),
        )
        self.y_mean_ = _safe_column_mean(y)
        self._response_scale_for_scoring_ = _training_response_scale(y)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            y -= self.y_mean_
        _require_finite_output(y, operation="Response centering")

        scale_y = self.scale if self.scale_y is None else self.scale_y

        if scale_y:
            self.y_scale_ = _safe_sample_scale(y)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                y /= self.y_scale_
            _require_finite_output(y, operation="Response scaling")
        else:
            self.y_scale_ = np.ones(y.shape[1], dtype=np.float64)

        X_cs = X
        y_cs = y
        _check_core_product_range(X_cs, y_cs)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            result = fit_pipls_core(
                X_cs,
                y_cs,
                predictor_rank=predictor_rank,
                n_components=self.n_components,
                response_subspace=self.response_subspace,
                svd_solver=self.svd_solver,
                random_state=self.random_state,
            )

        _require_finite_core_result(result)
        self.max_predictor_rank_ = max_predictor_rank
        self.decomposition_ = PiPLSDecomposition._from_core_result(result)
        self.x_rotations_ = self.decomposition_.predictor_directions
        self.y_rotations_ = self.decomposition_.response_directions
        self._n_features_out = self.n_components

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            coef_matrix = (
                result.standardized_regression_map
                * self.y_scale_[None, :]
                / self.x_scale_[:, None]
            )
            self.coef_ = np.asarray(coef_matrix.T, dtype=np.float64)
            self.intercept_ = self.y_mean_ - self.x_mean_ @ coef_matrix
            self.x_scores_ = X_cs @ self.x_rotations_
            self.y_scores_ = y_cs @ self.y_rotations_
        for name in ("coef_", "intercept_", "x_scores_", "y_scores_"):
            _require_finite_output(getattr(self, name), operation=f"Fitted {name}")
        x_loadings, _, _, _ = np.linalg.lstsq(self.x_scores_, X_cs, rcond=None)
        y_loadings, _, _, _ = np.linalg.lstsq(self.y_scores_, y_cs, rcond=None)
        self.x_loadings_ = np.asarray(x_loadings.T, dtype=np.float64)
        self.y_loadings_ = np.asarray(y_loadings.T, dtype=np.float64)
        _require_finite_output(self.x_loadings_, operation="Fitted x_loadings_")
        _require_finite_output(self.y_loadings_, operation="Fitted y_loadings_")

    def _validate_constructor_parameters(self) -> None:
        _as_positive_int(self.n_components, name="n_components")
        _as_positive_int(self.predictor_rank, name="predictor_rank")
        _validate_response_subspace(self.response_subspace)
        if self.n_components > self.predictor_rank:
            raise ValueError(
                "n_components must satisfy n_components <= predictor_rank; "
                f"got n_components={self.n_components}, "
                f"predictor_rank={self.predictor_rank}."
            )
        if not isinstance(self.scale, (bool, np.bool_)):
            raise ValueError(f"scale must be boolean; got {self.scale!r}.")
        _validate_optional_boolean(self.scale_x, name="scale_x")
        _validate_optional_boolean(self.scale_y, name="scale_y")
        if not isinstance(self.copy, (bool, np.bool_)):
            raise ValueError(f"copy must be boolean; got {self.copy!r}.")
        if not isinstance(self.svd_solver, str) or self.svd_solver not in (
            "full",
            "randomized",
            "auto",
        ):
            raise ValueError(
                f'svd_solver must be "full", "randomized", or "auto"; got {self.svd_solver!r}.'
            )
        _validate_random_state(self.random_state)


def _preprocess_predictor_block(
    X: FloatArray,
    *,
    scale: bool,
) -> tuple[FloatArray, FloatArray]:
    """Center and optionally scale one writable predictor training block."""

    x_mean = _safe_column_mean(X)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        X -= x_mean
    _require_finite_output(X, operation="Predictor centering")

    if scale:
        x_scale = _safe_sample_scale(X)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            X /= x_scale
        _require_finite_output(X, operation="Predictor scaling")
    else:
        x_scale = np.ones(X.shape[1], dtype=np.float64)
    return x_mean, x_scale


def _verified_predictor_rank(
    estimator: PiPLSRegression,
    X: ArrayLike,
    *,
    predictor_rank: int,
) -> int:
    """Return predictor rank verified after the estimator's X preprocessing."""

    probe = estimator.set_params(n_components=1, predictor_rank=predictor_rank)
    probe._validate_constructor_parameters()
    X_checked = check_array(
        X,
        ensure_2d=True,
        dtype=np.float64,
        ensure_min_samples=2,
        copy=True,
    )
    X_array = np.asarray(X_checked, dtype=np.float64)
    scale_x = probe.scale if probe.scale_x is None else probe.scale_x
    _preprocess_predictor_block(X_array, scale=bool(scale_x))

    _, verified_rank, _, _, _ = _decompose_predictors(
        X_array,
        predictor_rank=predictor_rank,
        svd_solver=probe.svd_solver,
        random_state=probe.random_state,
    )
    return verified_rank


def _clear_fitted_state(estimator: BaseEstimator) -> None:
    for name in tuple(vars(estimator)):
        if name.endswith("_") or name in {"_n_features_out", "_y_was_1d"}:
            delattr(estimator, name)


def _check_core_product_range(X: FloatArray, y: FloatArray) -> None:
    x_max = float(np.max(np.abs(X)))
    y_max = float(np.max(np.abs(y)))
    if x_max == 0.0 or y_max == 0.0:
        return
    log_bound = np.log(x_max) + np.log(y_max) + np.log(X.shape[0])
    if log_bound >= np.log(np.finfo(np.float64).max):
        raise FloatingPointError(
            "Pi-PLS cross-products may overflow; enable predictor and/or response scaling, "
            "or rescale the input data."
        )


def _require_finite_core_result(result: Any) -> None:
    for value in (result.Pi, result.C, result.W, result.P, result.D, result.Q):
        if not np.all(np.isfinite(value)):
            raise FloatingPointError(
                "Pi-PLS fitting produced nonfinite factorization values."
            )


def _require_finite_output(value: ArrayLike, *, operation: str) -> None:
    if not np.all(np.isfinite(np.asarray(value, dtype=np.float64))):
        raise FloatingPointError(
            f"{operation} produced values that are not representable as finite float64."
        )


def _validate_optional_boolean(value: object, *, name: str) -> None:
    if value is not None and not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be boolean or None; got {value!r}.")
