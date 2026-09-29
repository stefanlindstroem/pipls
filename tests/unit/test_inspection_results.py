from __future__ import annotations

import pickle

import numpy as np

from pipls.inspection import (
    BiplotCoordinates,
    LatentStructure,
    ObservationDiagnostics,
    PiPLSDisplayFactors,
    PredictionDiagnostics,
)


def _prediction_diagnostics() -> PredictionDiagnostics:
    return PredictionDiagnostics(
        observed=np.array([[1.0], [3.0], [5.0]], dtype=np.float32),
        predicted=np.array([[0.0], [4.0], [5.0]], dtype=np.float32),
        prediction_kind="external test predictions",
    )


def test_inspection_records_copy_arrays_and_preserve_pickle_immutability() -> None:
    x_scores = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    x_loadings = np.eye(2, dtype=np.float32)
    y_loadings = np.eye(2, dtype=np.float32)
    coefficients = np.eye(2, dtype=np.float32)
    structure = LatentStructure(x_scores, x_loadings, y_loadings, coefficients)
    x_scores[0, 0] = 9.0

    coordinates = BiplotCoordinates(
        sample_coordinates=np.eye(2, dtype=np.float32),
        predictor_coordinates=np.eye(2, dtype=np.float32),
        component_indices=np.array([0, 1], dtype=np.int64),
        scaling_factors=np.ones(2, dtype=np.float32),
    )
    observations = ObservationDiagnostics(
        score_distance=np.array([1.0, 2.0], dtype=np.float32),
        x_reconstruction_residual=np.array([0.1, 0.2], dtype=np.float32),
    )
    factors = PiPLSDisplayFactors(
        predictor_directions=np.eye(2, dtype=np.float32),
        dilation=np.array([2.0, 1.0], dtype=np.float32),
        response_directions=np.eye(2, dtype=np.float32),
    )
    predictions = _prediction_diagnostics()

    assert structure.x_scores[0, 0] == 1.0
    for record in (structure, coordinates, observations, factors, predictions):
        restored = pickle.loads(pickle.dumps(record))
        for value in vars(restored).values():
            if isinstance(value, np.ndarray):
                assert value.dtype in (np.dtype(np.float64), np.dtype(np.intp))
                assert not value.flags.writeable


def test_display_factors_derive_weighted_response_directions() -> None:
    factors = PiPLSDisplayFactors(
        predictor_directions=np.eye(2, dtype=np.float32),
        dilation=np.array([2.0, 1.0], dtype=np.float32),
        response_directions=np.eye(2, dtype=np.float32),
    )

    weighted = factors.weighted_response_directions
    np.testing.assert_array_equal(weighted, np.diag([2.0, 1.0]))
    assert not weighted.flags.writeable
    assert "weighted_response_directions" not in vars(factors)


def test_prediction_diagnostics_derive_dependent_fields() -> None:
    diagnostics = _prediction_diagnostics()

    np.testing.assert_array_equal(diagnostics.residual, [[1.0], [-1.0], [0.0]])
    np.testing.assert_array_equal(diagnostics.response_centers, [3.0])
    np.testing.assert_array_equal(diagnostics.response_scales, [2.0])
    np.testing.assert_allclose(diagnostics.observed_standardized, [[-1.0], [0.0], [1.0]])
    np.testing.assert_allclose(
        diagnostics.predicted_standardized,
        [[-1.5], [0.5], [1.0]],
    )
    np.testing.assert_allclose(
        diagnostics.residual_standardized,
        [[0.5], [-0.5], [0.0]],
    )
    np.testing.assert_allclose(diagnostics.standardized_rmse, [np.sqrt(1.0 / 6.0)])
    np.testing.assert_allclose(diagnostics.response_r2, [0.75])
