from __future__ import annotations

import pickle

import numpy as np
import pytest
from sklearn import config_context
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.metrics import r2_score
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    PredefinedSplit,
    RepeatedKFold,
    TimeSeriesSplit,
)

from pipls import PiPLSRegression, PiPLSSearchCV


def _data(n_samples: int = 18) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260719)
    X = rng.normal(size=(n_samples, 6))
    Y = X @ rng.normal(size=(6, 2)) + 0.05 * rng.normal(size=(n_samples, 2))
    return X, Y


def _fixed() -> PiPLSRegression:
    return PiPLSRegression(
        n_components=1,
        predictor_rank=2,
        scale=False,
        svd_solver="full",
        random_state=None,
    )


def test_ordered_oof_predictions_match_manual_fold_fits() -> None:
    X, Y = _data(12)
    splitter = KFold(n_splits=3, shuffle=False)
    search = PiPLSSearchCV(
        estimator=_fixed(),
        n_components_values=[1],
        predictor_rank_values=[2],
        max_predictor_rank=2,
        cv=splitter,
        n_jobs=1,
    ).fit(X, Y)

    expected = np.empty_like(Y)
    for train, validation in splitter.split(X, Y):
        expected[validation] = clone(_fixed()).fit(X[train], Y[train]).predict(X[validation])

    selection = search.select(rule="best_score")
    report = search.oof_report(X, Y, selection=selection)
    assert report.selection is selection
    np.testing.assert_allclose(report.oof_predictions, expected)
    np.testing.assert_array_equal(report.oof_prediction_counts, np.ones(X.shape[0]))
    assert report.selection.n_components == selection.n_components
    assert report.selection.predictor_rank == selection.predictor_rank
    assert report.has_complete_oof_coverage
    best_index = np.flatnonzero(
        (search.cv_results_["n_components"] == selection.n_components)
        & (search.cv_results_["predictor_rank"] == selection.predictor_rank)
    )
    assert best_index.size == 1
    assert report.selection.cv_mse_mean == pytest.approx(
        search.cv_results_["mean_response_standardized_mse"][int(best_index[0])]
    )
    assert report.pooled_oof_r2 == pytest.approx(
        r2_score(Y, expected, multioutput="uniform_average")
    )


def test_repeated_kfold_averages_predictions_and_records_counts() -> None:
    X, Y = _data()
    search = PiPLSSearchCV(
        estimator=_fixed(),
        n_components_values=[1],
        predictor_rank_values=[2],
        max_predictor_rank=2,
        cv=RepeatedKFold(n_splits=3, n_repeats=2, random_state=7),
        n_jobs=1,
    ).fit(X, Y)

    selection = search.select(n_components=1)
    report = search.oof_report(X, Y, selection=selection)
    np.testing.assert_array_equal(
        report.oof_prediction_counts,
        np.full(X.shape[0], 2, dtype=np.intp),
    )
    assert report.has_complete_oof_coverage
    assert np.all(np.isfinite(report.oof_predictions))


def test_predefined_and_temporal_splits_mark_uncovered_rows() -> None:
    X, Y = _data()
    predefined = PiPLSSearchCV(
        estimator=_fixed(),
        n_components_values=[1],
        predictor_rank_values=[2],
        max_predictor_rank=2,
        cv=PredefinedSplit(np.array([-1] * 9 + [0] * 9)),
    ).fit(X, Y)
    temporal = PiPLSSearchCV(
        estimator=_fixed(),
        n_components_values=[1],
        predictor_rank_values=[2],
        max_predictor_rank=2,
        cv=TimeSeriesSplit(n_splits=3),
    ).fit(X, Y)

    predefined_selection = predefined.select(n_components=1)
    temporal_selection = temporal.select(n_components=1)
    predefined_report = predefined.oof_report(
        X, Y, selection=predefined_selection
    )
    temporal_report = temporal.oof_report(X, Y, selection=temporal_selection)
    np.testing.assert_array_equal(predefined_report.oof_prediction_counts[:9], 0)
    assert np.isnan(predefined_report.oof_predictions[:9]).all()
    assert not predefined_report.has_complete_oof_coverage
    assert np.any(temporal_report.oof_prediction_counts == 0)
    uncovered = temporal_report.oof_prediction_counts == 0
    assert np.isnan(temporal_report.oof_predictions[uncovered]).all()
    assert not temporal_report.has_complete_oof_coverage


def test_grouped_splitters_are_supported_by_path_interface() -> None:
    X, Y = _data(24)
    groups = np.repeat(np.arange(8), 3)
    path = PiPLSSearchCV(
        estimator=_fixed(),
        n_components_values=[1],
        predictor_rank_values=[2],
        max_predictor_rank=2,
        cv=GroupKFold(n_splits=4),
    ).fit(X, Y, groups=groups)

    assert path.n_splits_ == 4


def test_groups_participate_in_path_metadata_routing() -> None:
    X, Y = _data(30)
    groups = np.repeat(np.arange(10), 3)
    estimator = PiPLSSearchCV(
        estimator=_fixed(),
        n_components_values=[1],
        predictor_rank_values=[2],
        max_predictor_rank=2,
        cv=GroupKFold(n_splits=3),
    )

    with config_context(enable_metadata_routing=True):
        routed = estimator.set_fit_request(groups=True)
        result = routed.fit(X, Y, groups=groups)

    assert result is routed
    assert result.n_splits_ == 3


@pytest.mark.parametrize("scoring", [None, "r2"])
def test_singleton_validation_rejects_foldwise_r2(scoring: object) -> None:
    X, Y = _data(10)
    indices = np.arange(X.shape[0], dtype=np.intp)
    splits = [
        (np.delete(indices, validation), np.asarray([validation], dtype=np.intp))
        for validation in indices
    ]
    estimator = PiPLSSearchCV(
        estimator=_fixed(),
        n_components_values=[1],
        predictor_rank_values=[2],
        max_predictor_rank=2,
        cv=splits,
        scoring=scoring,
    )

    with pytest.raises(ValueError, match="R2 scoring is undefined"):
        estimator.fit(X, Y)


def test_oof_arrays_are_read_only_and_one_dimensional_targets_stay_one_dimensional() -> None:
    X, Y = _data()
    y = Y[:, 0]
    search = PiPLSSearchCV(
        estimator=_fixed(),
        n_components_values=[1],
        predictor_rank_values=[2],
        max_predictor_rank=2,
        cv=3,
    ).fit(X, y)

    selection = search.select(n_components=1)
    report = search.oof_report(X, y, selection=selection)
    assert report.oof_predictions.shape == (X.shape[0],)
    assert not report.oof_predictions.flags.writeable
    assert not report.oof_prediction_counts.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        report.oof_predictions[0] = 0.0


class _SingleUseSplitter:
    def __init__(self, splits: list[tuple[np.ndarray, np.ndarray]]) -> None:
        self.splits = splits
        self.calls = 0

    def split(
        self,
        X: object,
        y: object | None = None,
        groups: object | None = None,
    ) -> object:
        del X, y, groups
        self.calls += 1
        if self.calls > 1:
            raise RuntimeError("split() was called more than once")
        return iter(self.splits)

    def get_n_splits(
        self,
        X: object = None,
        y: object | None = None,
        groups: object | None = None,
    ) -> int:
        del X, y, groups
        return len(self.splits)


def test_oof_report_reuses_defensive_read_only_search_splits() -> None:
    X, Y = _data()
    indices = np.arange(X.shape[0], dtype=np.intp)
    source_splits = [
        (indices[6:].copy(), indices[:6].copy()),
        (np.concatenate((indices[:6], indices[12:])), indices[6:12].copy()),
        (indices[:12].copy(), indices[12:].copy()),
    ]
    splitter = _SingleUseSplitter(source_splits)
    search = PiPLSSearchCV(
        estimator=_fixed(),
        n_components_values=[1],
        predictor_rank_values=[2],
        max_predictor_rank=2,
        cv=splitter,
        n_jobs=1,
    ).fit(X, Y)

    source_splits[0][0][0] = X.shape[0] + 10
    assert splitter.calls == 1
    assert all(
        not index.flags.writeable
        for split in search._cv_splits_
        for index in split
    )
    with pytest.raises(ValueError, match="read-only"):
        search._cv_splits_[0][0][0] = 0

    selection = search.select(n_components=1)
    report = search.oof_report(X, Y, selection=selection)

    assert splitter.calls == 1
    np.testing.assert_array_equal(report.oof_prediction_counts, np.ones(X.shape[0]))


def test_oof_report_requires_a_fitted_row_aligned_search_shape() -> None:
    X, Y = _data()
    fitted = PiPLSSearchCV(
        estimator=_fixed(),
        n_components_values=[1],
        predictor_rank_values=[2],
        max_predictor_rank=2,
        cv=3,
    ).fit(X, Y)
    selection = fitted.select(n_components=1)

    with pytest.raises(NotFittedError):
        PiPLSSearchCV().oof_report(X, Y, selection=selection)

    with pytest.raises(ValueError, match="same number of samples"):
        fitted.oof_report(X[:-1], Y[:-1], selection=selection)
    with pytest.raises(ValueError, match="same number of response columns"):
        fitted.oof_report(X, Y[:, :1], selection=selection)


def test_oof_report_does_not_mutate_search_state() -> None:
    X, Y = _data()
    search = PiPLSSearchCV(
        estimator=_fixed(),
        n_components_values=[1],
        predictor_rank_values=[2],
        max_predictor_rank=2,
        cv=3,
        n_jobs=1,
    ).fit(X, Y)
    before = pickle.dumps(search)

    selection = search.select(rule="best_score")
    search.oof_report(X, Y, selection=selection)

    assert pickle.dumps(search) == before
