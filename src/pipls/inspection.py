"""Pure numerical utilities for fitted-model inspection."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.utils.validation import check_is_fitted

from ._result_validation import _read_only_float_array, _read_only_int_array
from .decomposition import PiPLSDecomposition

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.intp]
PredictionKind: TypeAlias = Literal[
    "fitted values",
    "fixed-parameter OOF predictions",
    "selection-conditioned OOF predictions",
    "external test predictions",
]
"""Provenance label attached to prediction diagnostics."""


class _LatentStructureModel(Protocol):
    """Structural fitted-model contract used by shared PLS-family analysis."""

    x_scores_: ArrayLike
    x_loadings_: ArrayLike
    y_loadings_: ArrayLike
    coef_: ArrayLike

    def fit(self, X: ArrayLike, y: ArrayLike) -> Any:
        """Fit the estimator."""


class _ObservationModel(_LatentStructureModel, Protocol):
    """Latent-structure model with public X transform reconstruction methods."""

    def transform(self, X: ArrayLike) -> ArrayLike:
        """Transform predictor observations to X scores."""

    def inverse_transform(self, X: ArrayLike) -> ArrayLike:
        """Reconstruct predictor observations from X scores."""


_PREDICTION_KINDS: tuple[PredictionKind, ...] = (
    "fitted values",
    "fixed-parameter OOF predictions",
    "selection-conditioned OOF predictions",
    "external test predictions",
)
_PREDICTION_KIND_SET = frozenset(_PREDICTION_KINDS)

__all__ = [
    "BiplotCoordinates",
    "LatentStructure",
    "ObservationDiagnostics",
    "PiPLSDisplayFactors",
    "PredictionDiagnostics",
    "PredictionKind",
    "pipls_display_factors",
    "biplot_coordinates",
    "latent_structure",
    "observation_diagnostics",
    "prediction_diagnostics",
]


@dataclass(frozen=True)
class BiplotCoordinates:
    r"""Immutable balanced coordinates for a two-component PLS biplot.

    ``sample_coordinates`` and ``predictor_coordinates`` preserve the selected
    score-loading reconstruction while giving both coordinate sets equal
    Euclidean norm within each component. Arrays are defensive read-only copies;
    instances are normally obtained from :func:`biplot_coordinates`.

    Attributes
    ----------
    sample_coordinates : ndarray of shape (n_samples, 2)
        Balanced X-score coordinates for the observations.
    predictor_coordinates : ndarray of shape (n_features, 2)
        Balanced X-loading coordinates for the predictors.
    component_indices : ndarray of shape (2,)
        Selected zero-based component indices.
    scaling_factors : ndarray of shape (2,)
        Positive factors applied to scores and inversely to loadings.
    """

    sample_coordinates: FloatArray
    predictor_coordinates: FloatArray
    component_indices: IntArray
    scaling_factors: FloatArray

    def __post_init__(self) -> None:
        """Store defensive read-only copies of coordinate arrays."""

        object.__setattr__(
            self,
            "sample_coordinates",
            _read_only_float_array(
                self.sample_coordinates, name="sample_coordinates", ndim=2
            ),
        )
        object.__setattr__(
            self,
            "predictor_coordinates",
            _read_only_float_array(
                self.predictor_coordinates, name="predictor_coordinates", ndim=2
            ),
        )
        object.__setattr__(
            self,
            "component_indices",
            _read_only_int_array(self.component_indices, name="component_indices"),
        )
        object.__setattr__(
            self,
            "scaling_factors",
            _read_only_float_array(self.scaling_factors, name="scaling_factors"),
        )

    def __reduce__(self) -> tuple[type[BiplotCoordinates], tuple[object, ...]]:
        """Reconstruct so unpickled arrays remain read-only."""

        return (
            type(self),
            (
                self.sample_coordinates,
                self.predictor_coordinates,
                self.component_indices,
                self.scaling_factors,
            ),
        )


@dataclass(frozen=True)
class LatentStructure:
    r"""Immutable copies of public fitted PLS-family quantities.

    The coefficient orientation follows ``PLSRegression`` and
    ``PiPLSRegression``. Arrays are defensive read-only copies; instances are
    normally obtained from :func:`latent_structure`.

    Attributes
    ----------
    x_scores : ndarray of shape (n_samples, n_components)
        Fitted X-score matrix.
    x_loadings : ndarray of shape (n_features, n_components)
        X-loading matrix.
    y_loadings : ndarray of shape (n_targets, n_components)
        Y-loading matrix.
    coefficients : ndarray of shape (n_targets, n_features)
        Regression coefficients mapping predictors to responses.
    """

    x_scores: FloatArray
    x_loadings: FloatArray
    y_loadings: FloatArray
    coefficients: FloatArray

    def __post_init__(self) -> None:
        """Store defensive read-only copies of fitted-model arrays."""

        object.__setattr__(
            self,
            "x_scores",
            _read_only_float_array(self.x_scores, name="x_scores", ndim=2),
        )
        object.__setattr__(
            self,
            "x_loadings",
            _read_only_float_array(self.x_loadings, name="x_loadings", ndim=2),
        )
        object.__setattr__(
            self,
            "y_loadings",
            _read_only_float_array(self.y_loadings, name="y_loadings", ndim=2),
        )
        object.__setattr__(
            self,
            "coefficients",
            _read_only_float_array(self.coefficients, name="coefficients", ndim=2),
        )

    def __reduce__(self) -> tuple[type[LatentStructure], tuple[object, ...]]:
        """Reconstruct so unpickled arrays remain read-only."""

        return (
            type(self),
            (self.x_scores, self.x_loadings, self.y_loadings, self.coefficients),
        )

    @property
    def n_components(self) -> int:
        """Number of retained latent components."""

        return int(self.x_scores.shape[1])


@dataclass(frozen=True)
class ObservationDiagnostics:
    r"""Immutable raw PLS-family observation diagnostics.

    ``score_distance`` is the squared Mahalanobis distance from the fitted
    training-score center, using the Moore--Penrose inverse of the fitted
    training-score covariance. ``x_reconstruction_residual`` is the row-wise
    squared Euclidean residual after the public transform/inverse-transform round
    trip. No theoretical warning limits are attached. Returned arrays are
    defensive read-only copies.

    Attributes
    ----------
    score_distance : ndarray of shape (n_samples,)
        Raw squared score distances for the supplied observations.
    x_reconstruction_residual : ndarray of shape (n_samples,)
        Raw squared X-reconstruction residuals.
    """

    score_distance: FloatArray
    x_reconstruction_residual: FloatArray

    def __post_init__(self) -> None:
        """Store defensive read-only copies of observation diagnostics."""

        object.__setattr__(
            self,
            "score_distance",
            _read_only_float_array(self.score_distance, name="score_distance"),
        )
        object.__setattr__(
            self,
            "x_reconstruction_residual",
            _read_only_float_array(
                self.x_reconstruction_residual, name="x_reconstruction_residual"
            ),
        )

    def __reduce__(self) -> tuple[type[ObservationDiagnostics], tuple[object, ...]]:
        """Reconstruct so unpickled arrays remain read-only."""

        return type(self), (self.score_distance, self.x_reconstruction_residual)


@dataclass(frozen=True)
class PiPLSDisplayFactors:
    r"""Immutable display-oriented copy of a Π-PLS factorization.

    The predictor and response direction columns use one chosen deterministic
    display sign per paired latent mode. Applying the same sign to both sides
    preserves the centered/scaled regression map
    $\mathbf{P}\mathbf{D}\mathbf{Q}^{\mathsf T}$. The factor arrays are
    stored as defensive read-only copies. Weighted response directions are derived
    from the stored response directions and dilation.

    Attributes
    ----------
    predictor_directions : ndarray of shape (n_features, n_components)
        Display-signed copy of $\mathbf{P}$.
    dilation : ndarray of shape (n_components,)
        Diagonal values of $\mathbf{D}$.
    response_directions : ndarray of shape (n_targets, n_components)
        Display-signed copy of $\mathbf{Q}$.
    weighted_response_directions : ndarray of shape (n_targets, n_components)
        Derived read-only columns $D_k Q_{:k}$, equal to the corresponding
        columns of $\mathbf{Q}\mathbf{D}$ and to
        ``response_directions * dilation``.
    """

    predictor_directions: FloatArray
    dilation: FloatArray
    response_directions: FloatArray

    def __post_init__(self) -> None:
        """Store defensive read-only copies of display factors."""

        object.__setattr__(
            self,
            "predictor_directions",
            _read_only_float_array(
                self.predictor_directions, name="predictor_directions", ndim=2
            ),
        )
        object.__setattr__(
            self,
            "dilation",
            _read_only_float_array(self.dilation, name="dilation"),
        )
        object.__setattr__(
            self,
            "response_directions",
            _read_only_float_array(
                self.response_directions, name="response_directions", ndim=2
            ),
        )

    def __reduce__(self) -> tuple[type[PiPLSDisplayFactors], tuple[object, ...]]:
        """Reconstruct so unpickled arrays remain read-only."""

        return (
            type(self),
            (
                self.predictor_directions,
                self.dilation,
                self.response_directions,
            ),
        )

    @property
    def weighted_response_directions(self) -> FloatArray:
        r"""Derived read-only columns $D_k Q_{:k}$ of $\mathbf{Q}\mathbf{D}$."""

        weighted = _finite_product(
            self.response_directions,
            self.dilation[None, :],
            name="weighted_response_directions",
        )
        weighted.setflags(write=False)
        return weighted

    @property
    def n_components(self) -> int:
        """Number of retained Π-PLS paired latent modes."""

        return int(self.dilation.shape[0])


@dataclass(frozen=True)
class PredictionDiagnostics:
    r"""Immutable standardized prediction and residual diagnostics.

    All response matrices are two-dimensional, including single-response input.
    Centers and sample standard deviations are estimated from ``observed`` and
    applied unchanged to ``predicted``. Diagnostic arrays are derived once from
    the independent observed and predicted values and stored read-only.

    Attributes
    ----------
    observed, predicted, residual : ndarray of shape (n_samples, n_targets)
        Responses on their original scale, with residual defined as observed minus
        predicted.
    observed_standardized : ndarray of shape (n_samples, n_targets)
        Standardized observed responses.
    predicted_standardized : ndarray of shape (n_samples, n_targets)
        Standardized predicted responses.
    residual_standardized : ndarray of shape (n_samples, n_targets)
        Standardized residuals.
    response_centers, response_scales : ndarray of shape (n_targets,)
        Display centering and sample-standard-deviation vectors from the observed
        responses.
    standardized_rmse : ndarray of shape (n_targets,)
        Response-wise root mean squared standardized residual.
    response_r2 : ndarray of shape (n_targets,)
        Response-wise coefficient of determination for the supplied predictions.
    prediction_kind : PredictionKind
        Explicit provenance of the supplied predictions.
    """

    observed: FloatArray
    predicted: FloatArray
    prediction_kind: PredictionKind
    residual: FloatArray = field(init=False)
    observed_standardized: FloatArray = field(init=False)
    predicted_standardized: FloatArray = field(init=False)
    residual_standardized: FloatArray = field(init=False)
    response_centers: FloatArray = field(init=False)
    response_scales: FloatArray = field(init=False)
    standardized_rmse: FloatArray = field(init=False)
    response_r2: FloatArray = field(init=False)

    def __post_init__(self) -> None:
        observed = _read_only_float_array(self.observed, name="observed", ndim=2)
        predicted = _read_only_float_array(self.predicted, name="predicted", ndim=2)
        prediction_kind = self.prediction_kind

        if observed.shape[0] < 2 or observed.shape[1] == 0:
            raise ValueError(
                "Prediction diagnostics require at least two observations and one response."
            )
        if predicted.shape != observed.shape:
            raise ValueError("predicted must have the same shape as observed.")

        response_centers = _safe_column_mean(observed, name="response_centers")
        response_scales = _safe_sample_scales(
            observed,
            response_centers,
            name="response_scales",
        )
        constant = np.flatnonzero(response_scales == 0.0)
        if constant.size:
            columns = ", ".join(str(int(index)) for index in constant)
            raise ValueError(
                f"observed contains constant response columns at indices: {columns}."
            )

        residual = _finite_difference(observed, predicted, name="residual")
        observed_standardized = _finite_divide(
            _finite_difference(
                observed,
                response_centers[None, :],
                name="centered observed responses",
            ),
            response_scales[None, :],
            name="observed_standardized",
        )
        predicted_standardized = _finite_divide(
            _finite_difference(
                predicted,
                response_centers[None, :],
                name="centered predicted responses",
            ),
            response_scales[None, :],
            name="predicted_standardized",
        )
        residual_standardized = _finite_divide(
            residual,
            response_scales[None, :],
            name="residual_standardized",
        )
        standardized_rmse = _safe_column_rmse(
            residual_standardized,
            name="standardized_rmse",
        )
        response_r2 = _safe_response_r2(
            standardized_rmse,
            n_samples=observed.shape[0],
        )

        for values in (
            residual,
            observed_standardized,
            predicted_standardized,
            residual_standardized,
            response_centers,
            response_scales,
            standardized_rmse,
            response_r2,
        ):
            values.setflags(write=False)

        object.__setattr__(self, "observed", observed)
        object.__setattr__(self, "predicted", predicted)
        object.__setattr__(self, "prediction_kind", prediction_kind)
        object.__setattr__(self, "residual", residual)
        object.__setattr__(self, "observed_standardized", observed_standardized)
        object.__setattr__(self, "predicted_standardized", predicted_standardized)
        object.__setattr__(self, "residual_standardized", residual_standardized)
        object.__setattr__(self, "response_centers", response_centers)
        object.__setattr__(self, "response_scales", response_scales)
        object.__setattr__(self, "standardized_rmse", standardized_rmse)
        object.__setattr__(self, "response_r2", response_r2)

    def __reduce__(self) -> tuple[type[PredictionDiagnostics], tuple[object, ...]]:
        """Reconstruct so unpickled arrays remain read-only."""

        return type(self), (self.observed, self.predicted, self.prediction_kind)


def biplot_coordinates(
    structure: LatentStructure,
    *,
    components: Sequence[int] = (0, 1),
) -> BiplotCoordinates:
    r"""Return balanced coordinates for a two-component PLS score-loading biplot.

    For selected score and X-loading columns $t_k$ and $p_k$, define

    \begin{equation}
    a_k
    =
    \sqrt{\frac{\lVert p_k \rVert_2}{\lVert t_k \rVert_2}},
    \qquad
    \widetilde t_k=a_kt_k,
    \qquad
    \widetilde p_k=\frac{p_k}{a_k}.
    \end{equation}

    The balanced coordinates preserve
    $\widetilde{\mathbf{T}}\widetilde{\mathbf{P}}^{\mathsf T}
    =\mathbf{T}_{\mathcal K}\mathbf{P}_{\mathcal K}^{\mathsf T}$ and
    have equal score and loading norm within each selected component.

    Parameters
    ----------
    structure : LatentStructure
        Extracted fitted PLS-family quantities.
    components : sequence of int, default=(0, 1)
        Exactly two distinct zero-based component indices.

    Returns
    -------
    BiplotCoordinates
        Read-only balanced sample and predictor coordinates.
    """

    if not isinstance(structure, LatentStructure):
        raise TypeError("structure must be a LatentStructure instance.")
    indices = _two_component_indices(components, size=structure.n_components)
    selected_scores = structure.x_scores[:, indices]
    selected_loadings = structure.x_loadings[:, indices]
    scaling_factors = _safe_biplot_scaling_factors(
        selected_scores,
        selected_loadings,
    )
    sample_coordinates = _finite_product(
        selected_scores,
        scaling_factors[None, :],
        name="balanced sample coordinates",
    )
    predictor_coordinates = _finite_divide(
        selected_loadings,
        scaling_factors[None, :],
        name="balanced predictor coordinates",
    )

    return BiplotCoordinates(
        sample_coordinates=sample_coordinates,
        predictor_coordinates=predictor_coordinates,
        component_indices=np.array(indices, dtype=np.int64),
        scaling_factors=scaling_factors,
    )


def latent_structure(model: object) -> LatentStructure:
    r"""Return defensive copies of public fitted PLS-family arrays.

    The function uses a structural fitted-model contract rather than a concrete
    estimator class. Compatible fitted models expose ``x_scores_``,
    ``x_loadings_``, ``y_loadings_``, and ``coef_`` with the usual PLS-family
    orientations.

    Parameters
    ----------
    model : object
        Fitted PLS-family estimator, such as :class:`pipls.PiPLSRegression` or
        scikit-learn ``PLSRegression``.

    Returns
    -------
    LatentStructure
        Read-only scores, loadings, and coefficients.
    """

    fitted = cast(
        _LatentStructureModel,
        _require_fitted_model(
            model,
            attributes=("x_scores_", "x_loadings_", "y_loadings_", "coef_"),
        ),
    )

    x_scores = _finite_matrix(fitted.x_scores_, name="model.x_scores_")
    x_loadings = _finite_matrix(fitted.x_loadings_, name="model.x_loadings_")
    y_loadings = _finite_matrix(fitted.y_loadings_, name="model.y_loadings_")
    coefficients = _finite_matrix(fitted.coef_, name="model.coef_")

    n_components = x_scores.shape[1]
    if n_components == 0:
        raise ValueError("model must contain at least one retained component.")
    if x_loadings.shape[1] != n_components:
        raise ValueError(
            "model.x_scores_ and model.x_loadings_ must contain the same number of components."
        )
    if y_loadings.shape[1] != n_components:
        raise ValueError(
            "model.x_scores_ and model.y_loadings_ must contain the same number of components."
        )

    expected_coefficients = (y_loadings.shape[0], x_loadings.shape[0])
    if coefficients.shape != expected_coefficients:
        raise ValueError(
            "model.coef_ must have shape (n_targets, n_features): "
            f"expected {expected_coefficients}, got {coefficients.shape}."
        )

    return LatentStructure(
        x_scores=x_scores,
        x_loadings=x_loadings,
        y_loadings=y_loadings,
        coefficients=coefficients,
    )


def observation_diagnostics(
    model: object,
    X: ArrayLike,
) -> ObservationDiagnostics:
    r"""Return raw score-distance and X-reconstruction diagnostics.

    The fitted training scores establish the score center and covariance. The
    supplied observations are transformed with the fitted model, and the covariance
    inverse is calculated with :func:`numpy.linalg.pinv`. No theoretical warning
    limits are calculated.

    Parameters
    ----------
    model : object
        Fitted PLS-family estimator exposing public X scores, X loadings,
        ``transform()``, and ``inverse_transform()``.
    X : array-like of shape (n_samples, n_features)
        Predictor observations on the fitted model's input scale.

    Returns
    -------
    ObservationDiagnostics
        Read-only raw score distances and squared X-reconstruction residuals.
    """

    fitted = cast(
        _ObservationModel,
        _require_fitted_model(
            model,
            attributes=("x_scores_", "x_loadings_"),
            methods=("transform", "inverse_transform"),
        ),
    )

    training_scores = _finite_matrix(fitted.x_scores_, name="model.x_scores_")
    x_loadings = _finite_matrix(fitted.x_loadings_, name="model.x_loadings_")
    X_values = _finite_matrix(X, name="X")
    if training_scores.shape[0] < 2:
        raise ValueError("model.x_scores_ must contain at least two training observations.")
    if X_values.shape[1] != x_loadings.shape[0]:
        raise ValueError(
            "X must contain the fitted number of predictor columns: "
            f"expected {x_loadings.shape[0]}, got {X_values.shape[1]}."
        )

    score_center = _safe_column_mean(training_scores, name="training-score center")
    centered_training_scores = _finite_difference(
        training_scores,
        score_center[None, :],
        name="centered training scores",
    )
    covariance_scale = float(np.max(np.abs(centered_training_scores)))
    if covariance_scale == 0.0:
        scaled_training_scores = centered_training_scores
    else:
        scaled_training_scores = centered_training_scores / covariance_scale
    score_covariance = (
        scaled_training_scores.T @ scaled_training_scores
    ) / (training_scores.shape[0] - 1)
    inverse_covariance = _finite_derived_array(
        np.linalg.pinv(score_covariance),
        name="training-score covariance pseudoinverse",
    )

    transformed = _finite_matrix(fitted.transform(X), name="model.transform(X)")
    if transformed.shape != (X_values.shape[0], training_scores.shape[1]):
        raise ValueError(
            "model.transform(X) must have shape (n_samples, n_components): "
            f"expected {(X_values.shape[0], training_scores.shape[1])}, "
            f"got {transformed.shape}."
        )
    centered_scores = _finite_difference(
        transformed,
        score_center[None, :],
        name="centered transformed scores",
    )
    scaled_scores = (
        centered_scores if covariance_scale == 0.0 else centered_scores / covariance_scale
    )
    with np.errstate(over="ignore", invalid="ignore"):
        score_distance = np.einsum(
            "ij,jk,ik->i",
            scaled_scores,
            inverse_covariance,
            scaled_scores,
        )
    score_distance = _finite_derived_array(score_distance, name="score_distance")
    score_distance = np.maximum(score_distance, 0.0)
    reconstructed = _finite_matrix(
        fitted.inverse_transform(transformed),
        name="model.inverse_transform(model.transform(X))",
    )
    reconstruction_error = _finite_difference(
        X_values,
        reconstructed,
        name="X reconstruction error",
    )
    x_reconstruction_residual = _safe_squared_row_norms(
        reconstruction_error,
        name="x_reconstruction_residual",
    )

    return ObservationDiagnostics(
        score_distance=score_distance,
        x_reconstruction_residual=x_reconstruction_residual,
    )


def _predictor_display_signs(predictor_directions: FloatArray) -> NDArray[np.int8]:
    """Return predictor-anchored component signs without modifying the input."""

    n_components = predictor_directions.shape[1]
    component_signs = np.ones(n_components, dtype=np.int8)
    pivots = np.argmax(np.abs(predictor_directions), axis=0)
    components = np.arange(n_components)
    component_signs[predictor_directions[pivots, components] < 0.0] = -1
    return component_signs


def pipls_display_factors(
    decomposition: PiPLSDecomposition,
    *,
    response_index: int | None = None,
    response_sign: Literal["positive", "negative"] = "positive",
) -> PiPLSDisplayFactors:
    r"""Return copied Π-PLS factors with deterministic display signs.

    By default, the first largest-magnitude entry of each predictor direction is
    made nonnegative. When ``response_index`` is supplied, each component is
    instead oriented so that the selected response-direction entry has the
    requested nonnegative or nonpositive sign. An exactly zero selected entry
    uses the default predictor-based sign for that component. The same sign is
    applied to the paired predictor and response directions, preserving
    $\mathbf{P}\mathbf{D}\mathbf{Q}^{\mathsf T}$.

    Parameters
    ----------
    decomposition : pipls.decomposition.PiPLSDecomposition
        Public fitted Π-PLS decomposition.
    response_index : int or None, default=None
        Zero-based response row used to orient every component. ``None`` uses the
        default predictor-based convention.
    response_sign : {"positive", "negative"}, default="positive"
        Requested sign for the selected response row. ``"positive"`` means
        nonnegative and ``"negative"`` means nonpositive. A negative response
        sign requires ``response_index``.

    Returns
    -------
    PiPLSDisplayFactors
        Read-only display factors.
    """

    if not isinstance(decomposition, PiPLSDecomposition):
        raise TypeError("decomposition must be a PiPLSDecomposition.")
    if response_sign not in ("positive", "negative"):
        raise ValueError("response_sign must be 'positive' or 'negative'.")
    if response_index is None and response_sign == "negative":
        raise ValueError("response_sign='negative' requires response_index.")
    if response_index is not None and (
        isinstance(response_index, (bool, np.bool_))
        or not isinstance(response_index, (int, np.integer))
    ):
        raise TypeError("response_index must be an integer or None.")

    predictor_directions = decomposition.predictor_directions.copy()
    response_directions = decomposition.response_directions.copy()
    dilation = decomposition.dilation.copy()

    component_signs = _predictor_display_signs(predictor_directions)
    if response_index is not None:
        resolved_response_index = int(response_index)
        n_targets = response_directions.shape[0]
        if not 0 <= resolved_response_index < n_targets:
            raise ValueError(
                "response_index must be between 0 and "
                f"{n_targets - 1}; got {resolved_response_index}."
            )
        response_anchor = response_directions[resolved_response_index]
        nonzero = response_anchor != 0.0
        desired_sign = 1.0 if response_sign == "positive" else -1.0
        component_signs[nonzero] = np.where(
            response_anchor[nonzero] * desired_sign < 0.0,
            -1,
            1,
        )

    predictor_directions *= component_signs[None, :]
    response_directions *= component_signs[None, :]
    _finite_product(
        response_directions,
        dilation[None, :],
        name="weighted_response_directions",
    )
    return PiPLSDisplayFactors(
        predictor_directions=predictor_directions,
        dilation=dilation,
        response_directions=response_directions,
    )


def prediction_diagnostics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    prediction_kind: PredictionKind,
) -> PredictionDiagnostics:
    r"""Return standardized observed, predicted, and residual diagnostics.

    Residuals use observed minus predicted. Response centers and scales are
    calculated from the observed responses. Scales are sample standard deviations
    with ``ddof=1`` and are applied unchanged to predictions.

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n_samples,) or (n_samples, n_targets)
        Aligned observed and predicted responses.
    prediction_kind : PredictionKind
        Explicit provenance label for the predictions.

    Returns
    -------
    PredictionDiagnostics
        Read-only response matrices, standardization statistics, response-wise
        standardized RMSE, and response-wise coefficient of determination.
    """

    if prediction_kind not in _PREDICTION_KIND_SET:
        choices = ", ".join(repr(choice) for choice in _PREDICTION_KINDS)
        raise ValueError(f"prediction_kind must be one of {choices}.")

    observed = _response_matrix(y_true, name="y_true")
    predicted = _response_matrix(y_pred, name="y_pred")
    if observed.shape != predicted.shape:
        raise ValueError(
            "y_true and y_pred must have identical shapes after response normalization: "
            f"got {observed.shape} and {predicted.shape}."
        )
    return PredictionDiagnostics(
        observed=observed,
        predicted=predicted,
        prediction_kind=prediction_kind,
    )


def _response_matrix(values: ArrayLike, *, name: str) -> FloatArray:
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


def _require_fitted_model(
    model: object,
    *,
    attributes: Sequence[str],
    methods: Sequence[str] = (),
) -> object:
    """Validate the structural contract and return the fitted model."""

    if not callable(getattr(model, "fit", None)):
        raise TypeError("model must be a PLS-family estimator with a callable fit method.")
    missing_methods = [name for name in methods if not callable(getattr(model, name, None))]
    if missing_methods:
        joined = ", ".join(f"{name}()" for name in missing_methods)
        raise TypeError(f"model must expose callable {joined} methods.")
    check_is_fitted(cast(Any, model), attributes=list(attributes))
    return model


def _finite_matrix(values: ArrayLike, *, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional; got shape {array.shape}.")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one row and one column.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _two_component_indices(components: Sequence[int], *, size: int) -> tuple[int, int]:
    values = tuple(components)
    if len(values) != 2:
        raise ValueError("components must contain exactly two indices.")
    if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in values):
        raise TypeError("components must contain integer indices.")
    first, second = (int(value) for value in values)
    if first == second:
        raise ValueError("components must contain two distinct indices.")
    if first < 0 or second < 0 or first >= size or second >= size:
        raise ValueError(f"components must lie in [0, {size - 1}]; got {(first, second)}.")
    return first, second


def _finite_derived_array(values: ArrayLike, *, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} cannot be represented as finite float64 values.")
    return np.array(array, dtype=np.float64, copy=True)


def _finite_difference(left: ArrayLike, right: ArrayLike, *, name: str) -> FloatArray:
    with np.errstate(over="ignore", invalid="ignore"):
        difference = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return _finite_derived_array(difference, name=name)


def _finite_product(left: ArrayLike, right: ArrayLike, *, name: str) -> FloatArray:
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        product = np.asarray(left, dtype=np.float64) * np.asarray(right, dtype=np.float64)
    return _finite_derived_array(product, name=name)


def _finite_divide(numerator: ArrayLike, denominator: ArrayLike, *, name: str) -> FloatArray:
    with np.errstate(divide="ignore", over="ignore", invalid="ignore", under="ignore"):
        quotient = np.asarray(numerator, dtype=np.float64) / np.asarray(
            denominator,
            dtype=np.float64,
        )
    return _finite_derived_array(quotient, name=name)


def _safe_column_mean(values: FloatArray, *, name: str) -> FloatArray:
    scales = np.max(np.abs(values), axis=0)
    means = np.zeros(values.shape[1], dtype=np.float64)
    nonzero = scales > 0.0
    if np.any(nonzero):
        normalized = values[:, nonzero] / scales[nonzero]
        means[nonzero] = np.mean(normalized, axis=0) * scales[nonzero]
    return _finite_derived_array(means, name=name)


def _safe_sample_scales(
    values: FloatArray,
    centers: FloatArray,
    *,
    name: str,
) -> FloatArray:
    centered = _finite_difference(
        values,
        centers[None, :],
        name=f"centered values for {name}",
    )
    maxima = np.max(np.abs(centered), axis=0)
    scales = np.zeros(values.shape[1], dtype=np.float64)
    nonzero = maxima > 0.0
    if np.any(nonzero):
        normalized = centered[:, nonzero] / maxima[nonzero]
        mean_squares = np.sum(normalized * normalized, axis=0) / (values.shape[0] - 1)
        scales[nonzero] = _finite_product(
            maxima[nonzero],
            np.sqrt(mean_squares),
            name=name,
        )
    return _finite_derived_array(scales, name=name)


def _safe_biplot_scaling_factors(
    scores: FloatArray,
    loadings: FloatArray,
) -> FloatArray:
    score_maxima = np.max(np.abs(scores), axis=0)
    loading_maxima = np.max(np.abs(loadings), axis=0)
    if np.any(score_maxima == 0.0):
        raise ValueError("Selected PLS score columns must have nonzero norm.")
    if np.any(loading_maxima == 0.0):
        raise ValueError("Selected PLS X-loading columns must have nonzero norm.")

    normalized_scores = scores / score_maxima[None, :]
    normalized_loadings = loadings / loading_maxima[None, :]
    score_norm_factors = np.sqrt(np.sum(normalized_scores * normalized_scores, axis=0))
    loading_norm_factors = np.sqrt(
        np.sum(normalized_loadings * normalized_loadings, axis=0)
    )
    log_factors = 0.5 * (
        np.log(loading_maxima)
        + np.log(loading_norm_factors)
        - np.log(score_maxima)
        - np.log(score_norm_factors)
    )
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        scaling_factors = np.exp(log_factors)
    scaling_factors = _finite_derived_array(
        scaling_factors,
        name="biplot scaling_factors",
    )
    if np.any(scaling_factors == 0.0):
        raise ValueError("biplot scaling_factors cannot be represented as positive float64 values.")
    return scaling_factors


def _safe_squared_row_norms(values: FloatArray, *, name: str) -> FloatArray:
    maxima = np.max(np.abs(values), axis=1)
    squared_norms = np.zeros(values.shape[0], dtype=np.float64)
    nonzero = maxima > 0.0
    if np.any(nonzero):
        normalized = values[nonzero, :] / maxima[nonzero, None]
        normalized_squares = np.sum(normalized * normalized, axis=1)
        limits = np.sqrt(np.finfo(np.float64).max / normalized_squares)
        if np.any(maxima[nonzero] > limits):
            raise ValueError(f"{name} cannot be represented as finite float64 values.")
        squared_norms[nonzero] = (
            maxima[nonzero] * maxima[nonzero] * normalized_squares
        )
    return _finite_derived_array(squared_norms, name=name)


def _safe_column_rmse(values: FloatArray, *, name: str) -> FloatArray:
    maxima = np.max(np.abs(values), axis=0)
    rmse = np.zeros(values.shape[1], dtype=np.float64)
    nonzero = maxima > 0.0
    if np.any(nonzero):
        normalized = values[:, nonzero] / maxima[nonzero]
        normalized_rmse = np.sqrt(np.mean(normalized * normalized, axis=0))
        rmse[nonzero] = _finite_product(
            maxima[nonzero],
            normalized_rmse,
            name=name,
        )
    return _finite_derived_array(rmse, name=name)


def _safe_response_r2(standardized_rmse: FloatArray, *, n_samples: int) -> FloatArray:
    factor = float(n_samples) / float(n_samples - 1)
    limit = np.sqrt(np.finfo(np.float64).max / factor)
    if np.any(standardized_rmse > limit):
        raise ValueError("response_r2 cannot be represented as finite float64 values.")
    squared_rmse = _finite_product(
        standardized_rmse,
        standardized_rmse,
        name="squared standardized_rmse for response_r2",
    )
    scaled_error = _finite_product(
        squared_rmse,
        factor,
        name="scaled error for response_r2",
    )
    return _finite_difference(1.0, scaled_error, name="response_r2")
