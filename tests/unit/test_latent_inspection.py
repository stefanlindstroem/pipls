from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.cross_decomposition import PLSRegression
from sklearn.exceptions import NotFittedError

from pipls import PiPLSRegression
from pipls.inspection import (
    BiplotCoordinates,
    LatentStructure,
    ObservationDiagnostics,
    biplot_coordinates,
    latent_structure,
    observation_diagnostics,
)

ModelKind = Literal["pls", "pipls"]


def _data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(441)
    X = rng.normal(size=(36, 7))
    Y = X[:, :3] @ np.array(
        [
            [1.0, -0.5, 0.2],
            [0.4, 0.8, -0.3],
            [-0.2, 0.3, 0.9],
        ]
    ) + 0.1 * rng.normal(size=(36, 3))
    return X, Y


def _fitted_model(kind: ModelKind) -> tuple[Any, np.ndarray]:
    X, Y = _data()
    if kind == "pls":
        model = PLSRegression(n_components=3, scale=True).fit(X, Y)
    else:
        model = PiPLSRegression(
            n_components=3,
            predictor_rank=5,
            scale=True,
            svd_solver="full",
        ).fit(X, Y)
    return model, X


def _unfitted_model(kind: ModelKind) -> Any:
    if kind == "pls":
        return PLSRegression(n_components=2)
    return PiPLSRegression(n_components=2, predictor_rank=3)


@pytest.mark.parametrize("kind", ["pls", "pipls"])
def test_latent_structure_copies_public_fitted_arrays(kind: ModelKind) -> None:
    model, _ = _fitted_model(kind)

    structure = latent_structure(model)

    assert isinstance(structure, LatentStructure)
    np.testing.assert_array_equal(structure.x_scores, model.x_scores_)
    np.testing.assert_array_equal(structure.x_loadings, model.x_loadings_)
    np.testing.assert_array_equal(structure.y_loadings, model.y_loadings_)
    np.testing.assert_array_equal(structure.coefficients, model.coef_)
    assert structure.x_scores.shape == (36, 3)
    assert structure.x_loadings.shape == (7, 3)
    assert structure.y_loadings.shape == (3, 3)
    assert structure.n_components == 3


@pytest.mark.parametrize("kind", ["pls", "pipls"])
def test_latent_structure_returns_defensive_read_only_arrays(kind: ModelKind) -> None:
    model, _ = _fitted_model(kind)
    structure = latent_structure(model)
    original_score = float(structure.x_scores[0, 0])

    for values in (
        structure.x_scores,
        structure.x_loadings,
        structure.y_loadings,
        structure.coefficients,
    ):
        assert not values.flags.writeable

    model.x_scores_[0, 0] = original_score + 10.0
    assert structure.x_scores[0, 0] == original_score

    with pytest.raises(ValueError, match="read-only"):
        structure.x_loadings[0, 0] = 0.0


@pytest.mark.parametrize("kind", ["pls", "pipls"])
def test_latent_structure_rejects_an_unfitted_estimator(kind: ModelKind) -> None:
    with pytest.raises(NotFittedError):
        latent_structure(_unfitted_model(kind))


def test_latent_structure_rejects_an_incompatible_object() -> None:
    with pytest.raises(TypeError, match="PLS-family estimator"):
        latent_structure(object())  # type: ignore[arg-type]


@pytest.mark.parametrize("kind", ["pls", "pipls"])
def test_latent_structure_rejects_inconsistent_coefficient_shape(kind: ModelKind) -> None:
    model, _ = _fitted_model(kind)
    model.coef_ = np.ones((model.y_loadings_.shape[0], model.x_loadings_.shape[0] - 1))

    with pytest.raises(ValueError, match=r"shape \(n_targets, n_features\)"):
        latent_structure(model)


@pytest.mark.parametrize("kind", ["pls", "pipls"])
def test_latent_structure_supports_one_response(kind: ModelKind) -> None:
    rng = np.random.default_rng(817)
    X = rng.normal(size=(24, 5))
    y = X[:, 0] - 0.5 * X[:, 1] + 0.1 * rng.normal(size=24)
    if kind == "pls":
        model = PLSRegression(n_components=1).fit(X, y)
    else:
        model = PiPLSRegression(
            n_components=1,
            predictor_rank=3,
            svd_solver="full",
        ).fit(X, y)

    structure = latent_structure(model)

    assert structure.y_loadings.shape == (1, 1)
    assert structure.coefficients.shape == (1, 5)
    assert structure.y_loadings.shape[0] == 1


@pytest.mark.parametrize("kind", ["pls", "pipls"])
def test_latent_structure_rejects_nonfinite_fitted_arrays(kind: ModelKind) -> None:
    model, _ = _fitted_model(kind)
    model.x_scores_[0, 0] = np.nan

    with pytest.raises(ValueError, match="finite values"):
        latent_structure(model)


@pytest.mark.parametrize("kind", ["pls", "pipls"])
def test_observation_diagnostics_match_public_method_calculation(kind: ModelKind) -> None:
    model, X = _fitted_model(kind)

    diagnostics = observation_diagnostics(model, X)

    training_scores = model.x_scores_
    center = np.mean(training_scores, axis=0)
    centered_training = training_scores - center
    covariance = centered_training.T @ centered_training / (len(training_scores) - 1)
    scores = np.asarray(model.transform(X), dtype=np.float64)
    centered_scores = scores - center
    expected_distance = np.einsum(
        "ij,jk,ik->i",
        centered_scores,
        np.linalg.pinv(covariance),
        centered_scores,
    )
    reconstructed = np.asarray(model.inverse_transform(scores), dtype=np.float64)
    expected_residual = np.sum((X - reconstructed) ** 2, axis=1)

    assert isinstance(diagnostics, ObservationDiagnostics)
    np.testing.assert_allclose(diagnostics.score_distance, expected_distance)
    np.testing.assert_allclose(
        diagnostics.x_reconstruction_residual,
        expected_residual,
    )
    assert diagnostics.score_distance.shape == (len(X),)
    assert not diagnostics.score_distance.flags.writeable
    assert not diagnostics.x_reconstruction_residual.flags.writeable


@pytest.mark.parametrize("kind", ["pls", "pipls"])
def test_observation_diagnostics_accept_external_observations(kind: ModelKind) -> None:
    model, X = _fitted_model(kind)

    diagnostics = observation_diagnostics(model, X[:5])

    assert diagnostics.score_distance.shape == (5,)
    assert diagnostics.x_reconstruction_residual.shape == (5,)


@pytest.mark.parametrize("kind", ["pls", "pipls"])
def test_observation_diagnostics_rejects_wrong_feature_count(kind: ModelKind) -> None:
    model, X = _fitted_model(kind)

    with pytest.raises(ValueError, match="fitted number of predictor columns"):
        observation_diagnostics(model, X[:, :-1])


@pytest.mark.parametrize("kind", ["pls", "pipls"])
def test_observation_diagnostics_rejects_unfitted_estimator(kind: ModelKind) -> None:
    with pytest.raises(NotFittedError):
        observation_diagnostics(_unfitted_model(kind), np.ones((4, 3)))


@pytest.mark.parametrize("kind", ["pls", "pipls"])
def test_biplot_coordinates_preserve_reconstruction_and_balance_norms(
    kind: ModelKind,
) -> None:
    model, _ = _fitted_model(kind)
    structure = latent_structure(model)
    scores_before = structure.x_scores.copy()
    loadings_before = structure.x_loadings.copy()

    coordinates = biplot_coordinates(structure, components=(0, 2))

    assert isinstance(coordinates, BiplotCoordinates)
    expected = structure.x_scores[:, [0, 2]] @ structure.x_loadings[:, [0, 2]].T
    reconstructed = coordinates.sample_coordinates @ coordinates.predictor_coordinates.T
    np.testing.assert_allclose(reconstructed, expected)
    np.testing.assert_allclose(
        np.linalg.norm(coordinates.sample_coordinates, axis=0),
        np.linalg.norm(coordinates.predictor_coordinates, axis=0),
    )
    np.testing.assert_array_equal(coordinates.component_indices, np.array([0, 2]))
    assert coordinates.sample_coordinates.shape[0] == structure.x_scores.shape[0]
    assert coordinates.predictor_coordinates.shape[0] == structure.x_loadings.shape[0]
    for values in (
        coordinates.sample_coordinates,
        coordinates.predictor_coordinates,
        coordinates.component_indices,
        coordinates.scaling_factors,
    ):
        assert not values.flags.writeable
    np.testing.assert_array_equal(structure.x_scores, scores_before)
    np.testing.assert_array_equal(structure.x_loadings, loadings_before)


@pytest.mark.parametrize(
    ("components", "message"),
    [
        ((0,), "exactly two"),
        ((0, 0), "distinct"),
        ((0, 3), "must lie"),
    ],
)
def test_biplot_coordinates_reject_invalid_component_selections(
    components: tuple[int, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        biplot_coordinates(_structure_for_biplot(), components=components)


def test_biplot_coordinates_reject_zero_norm_loadings() -> None:
    structure = _structure_for_biplot()
    x_loadings = structure.x_loadings.copy()
    x_loadings[:, 1] = 0.0
    invalid = LatentStructure(
        x_scores=structure.x_scores,
        x_loadings=x_loadings,
        y_loadings=structure.y_loadings,
        coefficients=structure.coefficients,
    )

    with pytest.raises(ValueError, match="X-loading columns"):
        biplot_coordinates(invalid, components=(0, 1))


def _structure_for_biplot() -> LatentStructure:
    model, _ = _fitted_model("pls")
    return latent_structure(model)


class _ExtremeObservationModel(BaseEstimator):
    def __init__(self, *, reconstruct_sign: float = 1.0) -> None:
        self.x_scores_ = np.array(
            [[-1.0e200, 0.0], [1.0e200, 0.0], [0.0, 1.0e200]],
            dtype=np.float64,
        )
        self.x_loadings_ = np.eye(2)
        self.y_loadings_ = np.eye(2)
        self.coef_ = np.eye(2)
        self._reconstruct_sign = reconstruct_sign

    def fit(self, X: object, y: object) -> _ExtremeObservationModel:
        del X, y
        return self

    def transform(self, X: object) -> np.ndarray:
        return np.asarray(X, dtype=np.float64)

    def inverse_transform(self, X: object) -> np.ndarray:
        return self._reconstruct_sign * np.asarray(X, dtype=np.float64)


def test_biplot_coordinates_handle_extreme_representable_norms() -> None:
    structure = LatentStructure(
        x_scores=np.array([[1.0e200, 0.0], [0.0, 1.0e200]]),
        x_loadings=np.array([[1.0e-200, 0.0], [0.0, 1.0e-200]]),
        y_loadings=np.eye(2),
        coefficients=np.eye(2),
    )

    with np.errstate(all="raise"):
        coordinates = biplot_coordinates(structure)

    assert np.all(np.isfinite(coordinates.sample_coordinates))
    assert np.all(np.isfinite(coordinates.predictor_coordinates))
    np.testing.assert_allclose(
        coordinates.sample_coordinates @ coordinates.predictor_coordinates.T,
        np.eye(2),
    )


def test_observation_diagnostics_handle_extreme_score_covariance() -> None:
    model = _ExtremeObservationModel()
    X = model.x_scores_.copy()

    with np.errstate(all="raise"):
        diagnostics = observation_diagnostics(model, X)

    assert np.all(np.isfinite(diagnostics.score_distance))
    np.testing.assert_array_equal(diagnostics.x_reconstruction_residual, np.zeros(3))


def test_observation_diagnostics_reject_unrepresentable_reconstruction_residuals() -> None:
    model = _ExtremeObservationModel(reconstruct_sign=-1.0)
    X = np.array([[np.finfo(np.float64).max, 0.0]])

    with pytest.raises(ValueError, match="X reconstruction error cannot be represented"):
        observation_diagnostics(model, X)
