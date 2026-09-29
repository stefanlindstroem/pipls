"""Private fold engine used by :class:`pipls.PiPLSSearchCV`."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from time import perf_counter
from typing import Any, cast

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import ArrayLike, NDArray
from sklearn.base import clone
from sklearn.utils import _safe_indexing

from ._model_selection import CVSplit
from .metrics import _response_scale_for_scoring, _response_standardized_mse

FloatArray = NDArray[np.float64]
Scorer = Callable[[Any, ArrayLike, ArrayLike], float]
WarningCategory = type[Warning]


@dataclass(frozen=True)
class _PiPLSCandidate:
    """One admissible Π-PLS hyperparameter pair."""

    n_components: int
    predictor_rank: int

    @property
    def key(self) -> tuple[int, int]:
        """Return the stable cache key for this candidate."""

        return self.n_components, self.predictor_rank


@dataclass(frozen=True)
class _PiPLSCandidateResult:
    """Fold-level results for one Π-PLS candidate."""

    split_scores: FloatArray
    split_response_standardized_mse: FloatArray
    split_fit_times: FloatArray
    split_score_times: FloatArray


CandidateCache = dict[tuple[int, int], _PiPLSCandidateResult]


def _evaluate_candidate_batch(
    *,
    candidates: Iterable[_PiPLSCandidate],
    cache: CandidateCache,
    template: Any,
    n_components_key: str,
    predictor_rank_key: str,
    scorer: Scorer | None,
    use_default_scoring: bool,
    X: ArrayLike,
    y: ArrayLike,
    splits: tuple[CVSplit, ...],
    n_jobs: int | None,
    ignored_warning_categories: tuple[WarningCategory, ...] = (),
) -> None:
    """Evaluate uncached candidates and update ``cache`` in input order."""

    pending: list[_PiPLSCandidate] = []
    seen: set[tuple[int, int]] = set()
    for candidate in candidates:
        if candidate.key in cache or candidate.key in seen:
            continue
        pending.append(candidate)
        seen.add(candidate.key)
    if not pending:
        return

    results = cast(
        list[_PiPLSCandidateResult],
        Parallel(n_jobs=n_jobs)(
            delayed(_evaluate_candidate)(
                candidate=candidate,
                template=template,
                n_components_key=n_components_key,
                predictor_rank_key=predictor_rank_key,
                scorer=scorer,
                use_default_scoring=use_default_scoring,
                X=X,
                y=y,
                splits=splits,
                ignored_warning_categories=ignored_warning_categories,
            )
            for candidate in pending
        ),
    )
    for candidate, result in zip(pending, results, strict=True):
        cache[candidate.key] = result


def _evaluate_candidate(
    *,
    candidate: _PiPLSCandidate,
    template: Any,
    n_components_key: str,
    predictor_rank_key: str,
    scorer: Scorer | None,
    use_default_scoring: bool,
    X: ArrayLike,
    y: ArrayLike,
    splits: tuple[CVSplit, ...],
    ignored_warning_categories: tuple[WarningCategory, ...],
) -> _PiPLSCandidateResult:
    """Evaluate one candidate on every materialized split."""

    split_scores = np.empty(len(splits), dtype=np.float64)
    split_mse = np.empty(len(splits), dtype=np.float64)
    split_fit_times = np.empty(len(splits), dtype=np.float64)
    split_score_times = np.empty(len(splits), dtype=np.float64)
    params = {
        n_components_key: candidate.n_components,
        predictor_rank_key: candidate.predictor_rank,
    }

    for split_index, (train, validation) in enumerate(splits):
        estimator = clone(template).set_params(**params)
        X_train = _safe_indexing(X, train)
        y_train = _safe_indexing(y, train)
        X_validation = _safe_indexing(X, validation)
        y_validation = _safe_indexing(y, validation)
        fit_started = perf_counter()
        _fit_with_ignored_warnings(
            estimator,
            X_train,
            y_train,
            ignored_warning_categories=ignored_warning_categories,
        )
        split_fit_times[split_index] = perf_counter() - fit_started

        score_started = perf_counter()
        prediction = estimator.predict(X_validation)
        mse = _response_standardized_mse(
            y_validation,
            prediction,
            _response_scale_for_scoring(estimator),
        )
        split_mse[split_index] = mse

        if use_default_scoring:
            split_scores[split_index] = -mse
        else:
            assert scorer is not None
            score = float(scorer(estimator, X_validation, y_validation))
            if not np.isfinite(score):
                raise ValueError(
                    "The scoring callable returned a nonfinite value for "
                    f"n_components={candidate.n_components}, "
                    f"predictor_rank={candidate.predictor_rank}, split={split_index}."
                )
            split_scores[split_index] = score
        split_score_times[split_index] = perf_counter() - score_started

    return _PiPLSCandidateResult(
        split_scores=split_scores,
        split_response_standardized_mse=split_mse,
        split_fit_times=split_fit_times,
        split_score_times=split_score_times,
    )


def _ordered_oof_predictions(
    *,
    candidate: _PiPLSCandidate,
    template: Any,
    n_components_key: str,
    predictor_rank_key: str,
    X: ArrayLike,
    y: ArrayLike,
    splits: tuple[CVSplit, ...],
    n_jobs: int | None,
    ignored_warning_categories: tuple[WarningCategory, ...] = (),
) -> tuple[FloatArray, NDArray[np.intp]]:
    """Fit one fixed candidate on each split and return ordered predictions.

    Rows validated more than once are averaged. Rows never used for validation are
    filled with NaN and have a zero entry in ``prediction_counts``.
    """

    params = {
        n_components_key: candidate.n_components,
        predictor_rank_key: candidate.predictor_rank,
    }
    split_results = cast(
        list[tuple[NDArray[np.intp], FloatArray]],
        Parallel(n_jobs=n_jobs)(
            delayed(_fit_predict_split)(
                template=template,
                params=params,
                X=X,
                y=y,
                train=train,
                validation=validation,
                ignored_warning_categories=ignored_warning_categories,
            )
            for train, validation in splits
        ),
    )

    y_array = np.asarray(y)
    n_samples = int(y_array.shape[0])
    n_targets = 1 if y_array.ndim == 1 else int(y_array.shape[1])
    prediction_sum = np.zeros((n_samples, n_targets), dtype=np.float64)
    prediction_counts = np.zeros(n_samples, dtype=np.intp)

    for validation, prediction in split_results:
        prediction_sum[validation] += prediction
        prediction_counts[validation] += 1

    predictions = np.full((n_samples, n_targets), np.nan, dtype=np.float64)
    covered = prediction_counts > 0
    predictions[covered] = prediction_sum[covered] / prediction_counts[covered, None]
    return predictions, prediction_counts


def _fit_predict_split(
    *,
    template: Any,
    params: dict[str, int],
    X: ArrayLike,
    y: ArrayLike,
    train: NDArray[np.intp],
    validation: NDArray[np.intp],
    ignored_warning_categories: tuple[WarningCategory, ...],
) -> tuple[NDArray[np.intp], FloatArray]:
    """Fit and predict one split for ordered OOF aggregation."""

    estimator = clone(template).set_params(**params)
    X_train = _safe_indexing(X, train)
    y_train = _safe_indexing(y, train)
    X_validation = _safe_indexing(X, validation)
    _fit_with_ignored_warnings(
        estimator,
        X_train,
        y_train,
        ignored_warning_categories=ignored_warning_categories,
    )
    prediction = np.asarray(estimator.predict(X_validation), dtype=np.float64)
    if prediction.ndim == 1:
        prediction = prediction.reshape(-1, 1)
    return validation.copy(), prediction


def _fit_with_ignored_warnings(
    estimator: Any,
    X: ArrayLike,
    y: ArrayLike,
    *,
    ignored_warning_categories: tuple[WarningCategory, ...] = (),
) -> None:
    """Fit one estimator while ignoring only caller-owned warning categories."""

    with warnings.catch_warnings():
        for category in ignored_warning_categories:
            warnings.simplefilter("ignore", category)
        estimator.fit(X, y)
