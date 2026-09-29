from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import pipls
import pipls.inspection as inspection
from pipls.decomposition import PiPLSDecomposition
from pipls.inspection import pipls_display_factors, prediction_diagnostics


def _decomposition(
    predictor_directions: np.ndarray,
    response_directions: np.ndarray,
    dilation: np.ndarray,
) -> PiPLSDecomposition:
    _, n_components = predictor_directions.shape
    return PiPLSDecomposition(
        predictor_directions=predictor_directions,
        dilation=dilation,
        response_directions=response_directions,
        predictor_numerical_rank=n_components,
        predictor_numerical_rank_is_exact=True,
        rank_tolerance=1e-12,
        predictor_svd_solver="full",
    )


def test_inspection_names_are_submodule_exports_only() -> None:
    expected = {
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
    }

    assert expected <= set(inspection.__all__)
    assert expected.isdisjoint(pipls.__all__)


def test_pipls_display_factors_use_first_largest_predictor_entry_for_sign() -> None:
    predictor_directions = np.array(
        [
            [-2.0, 0.0, 0.0],
            [2.0, -3.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )
    response_directions = np.array(
        [
            [0.5, -0.25, 1.0],
            [-1.5, 0.75, 0.0],
        ]
    )
    dilation = np.array([3.0, 2.0, 0.0])
    decomposition = _decomposition(predictor_directions, response_directions, dilation)

    factors = pipls_display_factors(decomposition)

    np.testing.assert_array_equal(
        factors.predictor_directions,
        predictor_directions * np.array([-1.0, -1.0, 1.0]),
    )
    np.testing.assert_array_equal(
        factors.response_directions,
        response_directions * np.array([-1.0, -1.0, 1.0]),
    )
    np.testing.assert_array_equal(
        factors.weighted_response_directions,
        factors.response_directions * dilation,
    )
    assert factors.predictor_directions.shape == (3, 3)
    assert factors.response_directions.shape == (2, 3)
    assert factors.n_components == 3


def test_pipls_display_factors_can_anchor_signs_to_a_response() -> None:
    predictor_directions = np.array(
        [
            [-4.0, 1.0, -3.0],
            [2.0, -2.0, 1.0],
        ]
    )
    response_directions = np.array(
        [
            [-2.0, 3.0, 0.0],
            [1.0, -1.0, 4.0],
        ]
    )
    dilation = np.array([3.0, 2.0, 1.0])
    decomposition = _decomposition(predictor_directions, response_directions, dilation)

    positive = pipls_display_factors(
        decomposition,
        response_index=0,
        response_sign="positive",
    )
    negative = pipls_display_factors(
        decomposition,
        response_index=np.int64(0),
        response_sign="negative",
    )

    positive_signs = np.array([-1.0, 1.0, -1.0])
    negative_signs = np.array([1.0, -1.0, -1.0])
    np.testing.assert_array_equal(
        positive.predictor_directions,
        predictor_directions * positive_signs,
    )
    np.testing.assert_array_equal(
        positive.response_directions,
        response_directions * positive_signs,
    )
    np.testing.assert_array_equal(
        negative.predictor_directions,
        predictor_directions * negative_signs,
    )
    np.testing.assert_array_equal(
        negative.response_directions,
        response_directions * negative_signs,
    )
    assert np.all(positive.response_directions[0] >= 0.0)
    assert np.all(negative.response_directions[0] <= 0.0)


def test_pipls_display_factors_validate_response_orientation() -> None:
    decomposition = _decomposition(np.eye(2), np.eye(2), np.ones(2))

    with pytest.raises(ValueError, match="response_sign must be"):
        pipls_display_factors(
            decomposition,
            response_index=0,
            response_sign="up",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="requires response_index"):
        pipls_display_factors(decomposition, response_sign="negative")
    with pytest.raises(TypeError, match="response_index must be an integer or None"):
        pipls_display_factors(decomposition, response_index=True)
    with pytest.raises(TypeError, match="response_index must be an integer or None"):
        pipls_display_factors(decomposition, response_index=0.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"between 0 and 1; got -1"):
        pipls_display_factors(decomposition, response_index=-1)
    with pytest.raises(ValueError, match=r"between 0 and 1; got 2"):
        pipls_display_factors(decomposition, response_index=2)


def test_pipls_display_factors_are_defensive_read_only_copies() -> None:
    predictor_directions = np.eye(2)
    response_directions = np.eye(2)
    dilation = np.array([2.0, 1.0])
    decomposition = _decomposition(predictor_directions, response_directions, dilation)

    factors = pipls_display_factors(decomposition)

    for values in (
        factors.predictor_directions,
        factors.dilation,
        factors.response_directions,
        factors.weighted_response_directions,
    ):
        assert not values.flags.writeable
    assert "weighted_response_directions" not in vars(factors)
    assert not np.shares_memory(factors.predictor_directions, decomposition.predictor_directions)
    assert not np.shares_memory(factors.response_directions, decomposition.response_directions)
    assert not np.shares_memory(factors.dilation, decomposition.dilation)

    predictor_directions[0, 0] = 7.0
    response_directions[0, 0] = 8.0
    dilation[0] = 9.0
    assert factors.predictor_directions[0, 0] == 1.0
    assert factors.response_directions[0, 0] == 1.0
    assert factors.dilation[0] == 2.0

    with pytest.raises(ValueError, match="read-only"):
        factors.predictor_directions[0, 0] = 0.0
    with pytest.raises(FrozenInstanceError):
        factors.dilation = np.ones(2)  # type: ignore[misc]


def test_pipls_display_factors_require_public_decomposition() -> None:
    with pytest.raises(TypeError, match="PiPLSDecomposition"):
        pipls_display_factors(object())  # type: ignore[arg-type]


def test_prediction_diagnostics_standardize_from_observed_responses() -> None:
    observed = np.array(
        [
            [1.0, 10.0],
            [3.0, 14.0],
            [5.0, 18.0],
        ]
    )
    predicted = np.array(
        [
            [0.0, 12.0],
            [4.0, 10.0],
            [5.0, 20.0],
        ]
    )

    diagnostics = prediction_diagnostics(
        observed,
        predicted,
        prediction_kind="external test predictions",
    )

    expected_center = np.array([3.0, 14.0])
    expected_scale = np.array([2.0, 4.0])
    expected_residual = observed - predicted
    np.testing.assert_allclose(diagnostics.response_centers, expected_center)
    np.testing.assert_allclose(diagnostics.response_scales, expected_scale)
    np.testing.assert_allclose(diagnostics.residual, expected_residual)
    np.testing.assert_allclose(
        diagnostics.observed_standardized,
        (observed - expected_center) / expected_scale,
    )
    np.testing.assert_allclose(
        diagnostics.predicted_standardized,
        (predicted - expected_center) / expected_scale,
    )
    np.testing.assert_allclose(
        diagnostics.residual_standardized,
        expected_residual / expected_scale,
    )
    expected_standardized_rmse = np.sqrt(
        np.mean(np.square(expected_residual / expected_scale), axis=0)
    )
    expected_r2 = 1.0 - np.sum(np.square(expected_residual), axis=0) / np.sum(
        np.square(observed - expected_center),
        axis=0,
    )
    np.testing.assert_allclose(diagnostics.standardized_rmse, expected_standardized_rmse)
    np.testing.assert_allclose(diagnostics.response_r2, expected_r2)
    assert diagnostics.prediction_kind == "external test predictions"
    assert diagnostics.observed.shape == (3, 2)


def test_prediction_diagnostics_normalize_vector_inputs_to_two_dimensions() -> None:
    diagnostics = prediction_diagnostics(
        np.array([1.0, 3.0, 5.0]),
        np.array([[0.0], [4.0], [5.0]]),
        prediction_kind="fitted values",
    )

    assert diagnostics.observed.shape == (3, 1)
    assert diagnostics.predicted.shape == (3, 1)
    assert diagnostics.response_centers.shape == (1,)
    assert diagnostics.standardized_rmse.shape == (1,)
    assert diagnostics.response_r2.shape == (1,)


def test_prediction_diagnostics_are_defensive_read_only_copies() -> None:
    observed = np.array([1.0, 2.0, 4.0])
    predicted = np.array([1.5, 2.5, 3.5])

    diagnostics = prediction_diagnostics(
        observed,
        predicted,
        prediction_kind="fixed-parameter OOF predictions",
    )

    for values in (
        diagnostics.observed,
        diagnostics.predicted,
        diagnostics.residual,
        diagnostics.observed_standardized,
        diagnostics.predicted_standardized,
        diagnostics.residual_standardized,
        diagnostics.response_centers,
        diagnostics.response_scales,
        diagnostics.standardized_rmse,
        diagnostics.response_r2,
    ):
        assert not values.flags.writeable

    observed[0] = 100.0
    predicted[0] = 100.0
    assert diagnostics.observed[0, 0] == 1.0
    assert diagnostics.predicted[0, 0] == 1.5

    with pytest.raises(ValueError, match="read-only"):
        diagnostics.residual[0, 0] = 0.0


@pytest.mark.parametrize(
    ("y_true", "y_pred", "message"),
    [
        (np.ones((3, 2)), np.ones((3, 1)), "identical shapes"),
        (np.ones((1, 1)), np.ones((1, 1)), "at least two observations"),
        (np.ones((3, 1)), np.zeros((3, 1)), "constant response columns"),
        (np.array([1.0, np.nan]), np.array([1.0, 2.0]), "finite values"),
        (np.ones((2, 1, 1)), np.ones((2, 1, 1)), "one- or two-dimensional"),
    ],
)
def test_prediction_diagnostics_reject_invalid_response_inputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        prediction_diagnostics(y_true, y_pred, prediction_kind="fitted values")


def test_prediction_diagnostics_reject_unknown_prediction_kind() -> None:
    with pytest.raises(ValueError, match="prediction_kind must be one of"):
        prediction_diagnostics(
            np.array([1.0, 2.0]),
            np.array([1.0, 2.0]),
            prediction_kind="cross-validated predictions",  # type: ignore[arg-type]
        )


def test_prediction_diagnostics_handles_representable_extreme_values() -> None:
    observed = np.array([-1.0e307, 1.0e307])
    predicted = np.zeros(2)

    with np.errstate(all="raise"):
        diagnostics = prediction_diagnostics(
            observed,
            predicted,
            prediction_kind="external test predictions",
        )

    assert np.all(np.isfinite(diagnostics.response_centers))
    assert np.all(np.isfinite(diagnostics.response_scales))
    assert np.all(np.isfinite(diagnostics.standardized_rmse))
    assert np.all(np.isfinite(diagnostics.response_r2))
    np.testing.assert_allclose(diagnostics.response_scales, [np.sqrt(2.0) * 1.0e307])
    np.testing.assert_allclose(diagnostics.response_r2, [0.0], atol=1e-15)


def test_prediction_diagnostics_rejects_unrepresentable_residuals() -> None:
    observed = np.array([np.finfo(np.float64).max, 0.0])
    predicted = np.array([-np.finfo(np.float64).max, 0.0])

    with pytest.raises(ValueError, match="residual cannot be represented"):
        prediction_diagnostics(
            observed,
            predicted,
            prediction_kind="external test predictions",
        )


def test_prediction_diagnostics_rejects_unrepresentable_response_r2() -> None:
    with pytest.raises(ValueError, match="response_r2 cannot be represented"):
        prediction_diagnostics(
            np.array([-1.0, 1.0]),
            np.array([1.0e308, -1.0e308]),
            prediction_kind="external test predictions",
        )


def test_pipls_display_factors_rejects_unrepresentable_weighting() -> None:
    decomposition = _decomposition(
        np.eye(2),
        np.array([[np.finfo(np.float64).max, 0.0], [0.0, 1.0]]),
        np.array([2.0, 1.0]),
    )

    with pytest.raises(ValueError, match="weighted_response_directions cannot be represented"):
        pipls_display_factors(decomposition)
