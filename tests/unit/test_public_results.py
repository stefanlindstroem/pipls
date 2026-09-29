from __future__ import annotations

import pickle
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from pipls.component_path import PiPLSSelection
from pipls.decomposition import PiPLSDecomposition
from pipls.validation import PiPLSOOFReport


def _selection() -> PiPLSSelection:
    return PiPLSSelection(
        n_components=2,
        predictor_rank=3,
        predictor_rank_policy="optimized",
        mean_test_score=-0.4,
        cv_mse_mean=0.4,
        cv_mse_std=0.1,
        n_splits=5,
    )


def test_selection_is_frozen_and_pickleable() -> None:
    result = _selection()

    assert result.rule is None
    assert result.reference_minimum is None
    assert result.relative_tolerance is None
    assert result.absolute_tolerance is None
    assert result.predictor_rank_evidence is None
    assert result.cv_mse_threshold is None
    with pytest.raises(FrozenInstanceError):
        result.n_components = 1  # type: ignore[misc]

    restored = pickle.loads(pickle.dumps(result))
    assert restored == result


def test_selection_records_tolerance_provenance() -> None:
    minimum = PiPLSSelection(
        n_components=3,
        predictor_rank=5,
        predictor_rank_policy="optimized",
        mean_test_score=-0.40,
        cv_mse_mean=0.40,
        cv_mse_std=0.08,
        n_splits=5,
    )
    selection = PiPLSSelection(
        n_components=2,
        predictor_rank=4,
        predictor_rank_policy="optimized",
        mean_test_score=-0.43,
        cv_mse_mean=0.43,
        cv_mse_std=0.10,
        n_splits=5,
        rule="minimum_cv_mse",
        reference_minimum=minimum,
        relative_tolerance=0.10,
        absolute_tolerance=np.inf,
    )

    assert selection.rule == "minimum_cv_mse"
    assert selection.reference_minimum is minimum
    assert selection.relative_tolerance == pytest.approx(0.10)
    assert np.isposinf(selection.absolute_tolerance)
    assert selection.cv_mse_threshold == pytest.approx(0.44)

    restored = pickle.loads(pickle.dumps(selection))
    assert restored == selection
    assert restored.reference_minimum == minimum
    assert restored.cv_mse_threshold == selection.cv_mse_threshold


def test_decomposition_makes_defensive_read_only_copies_and_preserves_pickle() -> None:
    predictor_directions = np.eye(3, 2, dtype=np.float32)
    dilation = np.array([2.0, 1.0], dtype=np.float32)
    response_directions = np.eye(2, dtype=np.float32)
    decomposition = PiPLSDecomposition(
        predictor_directions=predictor_directions,
        dilation=dilation,
        response_directions=response_directions,
        predictor_numerical_rank=3,
        predictor_numerical_rank_is_exact=True,
        rank_tolerance=1e-12,
        predictor_svd_solver="full",
    )
    predictor_directions[0, 0] = 9.0
    dilation[0] = 9.0
    response_directions[0, 0] = 9.0

    assert decomposition.predictor_directions.dtype == np.dtype(np.float64)
    assert decomposition.dilation.dtype == np.dtype(np.float64)
    assert decomposition.response_directions.dtype == np.dtype(np.float64)
    assert decomposition.predictor_directions[0, 0] == 1.0
    assert decomposition.dilation[0] == 2.0
    assert decomposition.response_directions[0, 0] == 1.0
    assert all(
        not array.flags.writeable
        for array in (
            decomposition.predictor_directions,
            decomposition.dilation,
            decomposition.response_directions,
            decomposition.standardized_regression_map,
        )
    )

    restored = pickle.loads(pickle.dumps(decomposition))
    assert isinstance(restored, PiPLSDecomposition)
    assert not restored.predictor_directions.flags.writeable
    np.testing.assert_array_equal(restored.dilation, decomposition.dilation)


def test_oof_report_freezes_arrays_and_preserves_pickle() -> None:
    selection = _selection()
    report = PiPLSOOFReport(
        selection=selection,
        oof_predictions=np.array([[1.0, 2.0], [np.nan, np.nan], [3.0, 4.0]]),
        oof_prediction_counts=np.array([1, 0, 2], dtype=np.int64),
        pooled_oof_r2=0.25,
    )

    assert report.selection == selection
    assert type(report.has_complete_oof_coverage) is bool
    assert not report.oof_predictions.flags.writeable
    assert not report.oof_prediction_counts.flags.writeable
    assert not report.has_complete_oof_coverage

    restored = pickle.loads(pickle.dumps(report))
    assert isinstance(restored, PiPLSOOFReport)
    assert restored.selection == report.selection
    assert not restored.oof_predictions.flags.writeable
    assert not restored.oof_prediction_counts.flags.writeable
    np.testing.assert_array_equal(
        restored.oof_prediction_counts,
        report.oof_prediction_counts,
    )
