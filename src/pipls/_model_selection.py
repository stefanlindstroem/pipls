"""Private split, scoring, and rank-search primitives for ``PiPLSSearchCV``."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import r2_score
from sklearn.model_selection import check_cv

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.intp]
BoolArray = NDArray[np.bool_]
CVSplit = tuple[IntArray, IntArray]

_NUMERICAL_TIE_RTOL = 1e-12
_NUMERICAL_TIE_ATOL = 1e-15
_ADAPTIVE_INITIAL_POINTS = 7
_ADAPTIVE_EXHAUSTIVE_THRESHOLD = 5


@dataclass(frozen=True)
class _MaterializedCV:
    """Reusable validated cross-validation splits."""

    splits: tuple[CVSplit, ...]
    n_train_min: int


@dataclass(frozen=True)
class _PredictorRankScoreSelection:
    """Exact-reference and tolerance-qualified predictor-rank score evidence."""

    reference_rank: int
    reference_score: float
    selected_rank: int
    selected_score: float
    score_threshold: float


def _hard_predictor_rank_limit(
    *,
    n_features: int,
    n_train_min: int,
) -> int:
    r"""Return the fold-dimensional predictor-rank feasibility bound.

    The bound is

    \begin{equation}
    \min\left(
    p,
    n_{\mathrm{train,min}}-1
    \right),
    \end{equation}

    where ``p`` is ``n_features`` and ``n_train_min`` is the smallest
    materialized training-fold size. Verified fold numerical rank is applied
    separately by the search preflight.
    """

    return min(n_features, n_train_min - 1)


def _epv_predictor_rank(
    *,
    n_features: int,
    n_samples: int,
    samples_per_predictor_rank: float,
) -> int:
    r"""Return the nominal events-per-variable-inspired predictor rank.

    The nominal rule is

    \begin{equation}
    \min\left(
    p,
    \left\lceil n/c \right\rceil
    \right),
    \end{equation}

    where ``p`` is ``n_features``, ``n`` is ``n_samples``, and ``c`` is
    ``samples_per_predictor_rank``. Fold-dimensional and numerical-rank
    feasibility are intentionally not part of this helper.
    """

    samples_per_rank = float(samples_per_predictor_rank)
    if samples_per_rank <= n_samples / n_features:
        return n_features
    rule_limit = math.ceil(n_samples / samples_per_rank)
    return min(n_features, rule_limit)


def _materialize_cv_splits(
    cv: object,
    X: ArrayLike,
    y: ArrayLike,
    *,
    groups: ArrayLike | None = None,
) -> _MaterializedCV:
    """Materialize one validated set of CV splits for reuse by all candidates."""

    X_array = np.asarray(X)
    y_array = np.asarray(y)
    if X_array.ndim != 2:
        raise ValueError(f"X must be two-dimensional; got shape {X_array.shape}.")
    if y_array.ndim not in (1, 2):
        raise ValueError(f"y must be one- or two-dimensional; got shape {y_array.shape}.")
    if X_array.shape[0] != y_array.shape[0]:
        raise ValueError(
            "X and y must contain the same number of samples: "
            f"got {X_array.shape[0]} and {y_array.shape[0]}."
        )

    splitter = check_cv(cv=cv, y=y_array, classifier=False)
    raw_splits = cast(
        Iterable[tuple[ArrayLike, ArrayLike]],
        splitter.split(X_array, y_array, groups),
    )
    splits: list[CVSplit] = []
    n_samples = int(X_array.shape[0])
    for split_index, (train, validation) in enumerate(raw_splits):
        train_index = _as_index_array(
            train,
            name=f"training indices for split {split_index}",
            n_samples=n_samples,
        )
        validation_index = _as_index_array(
            validation,
            name=f"validation indices for split {split_index}",
            n_samples=n_samples,
        )
        if np.intersect1d(train_index, validation_index).size:
            raise ValueError(f"Training and validation indices overlap in split {split_index}.")
        train_index = np.array(train_index, dtype=np.intp, copy=True)
        validation_index = np.array(validation_index, dtype=np.intp, copy=True)
        train_index.setflags(write=False)
        validation_index.setflags(write=False)
        splits.append((train_index, validation_index))

    if not splits:
        raise ValueError("Cross-validation must produce at least one split.")
    return _MaterializedCV(
        splits=tuple(splits),
        n_train_min=min(train.size for train, _ in splits),
    )


def _validate_singleton_validation_scoring(
    scoring: object,
    splits: tuple[CVSplit, ...],
) -> None:
    """Reject ordinary R2 scoring for singleton validation sets."""

    if not any(validation.size == 1 for _, validation in splits):
        return
    score_func = getattr(scoring, "_score_func", None)
    if scoring is None or scoring == "r2" or score_func is r2_score:
        raise ValueError(
            "R2 scoring is undefined when a validation split contains fewer "
            "than two observations. Use a scorer defined for singleton "
            "validation sets."
        )


def _pooled_oof_r2(
    y: ArrayLike,
    predictions: ArrayLike,
    counts: ArrayLike,
) -> float | None:
    """Return pooled R2 on rows with OOF coverage, or None if unavailable."""

    y_array = np.asarray(y, dtype=np.float64)
    prediction_array = np.asarray(predictions, dtype=np.float64)
    count_array = np.asarray(counts, dtype=np.intp)
    covered = count_array > 0
    if int(np.sum(covered)) < 2:
        return None
    if y_array.ndim == 1 and prediction_array.ndim == 2:
        prediction_array = prediction_array[:, 0]
    return float(
        r2_score(
            y_array[covered],
            prediction_array[covered],
            multioutput="uniform_average",
        )
    )


def _logarithmic_predictor_rank_values(
    *,
    lower: int,
    upper: int,
    n_values: int = _ADAPTIVE_INITIAL_POINTS,
) -> IntArray:
    """Return deterministic approximately logarithmic integer ranks including endpoints."""

    if lower == upper:
        return np.asarray([lower], dtype=np.intp)

    values = np.exp(np.linspace(math.log(lower), math.log(upper), n_values))
    rounded = np.rint(values).astype(np.intp)
    unique_values: IntArray = np.unique(
        np.concatenate(
            (
                np.asarray([lower], dtype=np.intp),
                rounded,
                np.asarray([upper], dtype=np.intp),
            )
        )
    )
    return unique_values


def _adaptive_refinement_interval(
    predictor_ranks: ArrayLike,
    mean_scores: ArrayLike,
) -> tuple[int, int]:
    """Return the evaluated-neighbor interval around the exact score optimum."""

    ranks = np.asarray(predictor_ranks)
    scores = np.asarray(mean_scores, dtype=np.float64)
    reference_score = float(np.max(scores))
    reference_mask = _tied_score_mask(scores, reference_score)
    selected_rank = int(np.min(ranks[reference_mask]))
    order = np.argsort(ranks)
    sorted_ranks = ranks[order]
    selected_index = int(np.flatnonzero(sorted_ranks == selected_rank)[0])
    lower_index = max(0, selected_index - 1)
    upper_index = min(sorted_ranks.size - 1, selected_index + 1)
    return int(sorted_ranks[lower_index]), int(sorted_ranks[upper_index])


def _refine_adaptive_reference(
    *,
    allowed_ranks: IntArray,
    evaluate: Callable[[IntArray], None],
    evaluated_scores: Callable[[], tuple[IntArray, FloatArray]],
) -> None:
    """Refine adaptive coverage around the exact evaluated score optimum."""

    interval = allowed_ranks
    while True:
        if interval.size <= _ADAPTIVE_EXHAUSTIVE_THRESHOLD:
            proposed = interval
        else:
            logarithmic = _logarithmic_predictor_rank_values(
                lower=int(interval[0]),
                upper=int(interval[-1]),
            )
            indices = np.abs(interval[:, None] - logarithmic[None, :]).argmin(axis=0)
            proposed = np.asarray(np.unique(interval[indices]), dtype=np.intp)

        evaluate(proposed)

        if interval.size <= _ADAPTIVE_EXHAUSTIVE_THRESHOLD:
            return

        ranks, scores = evaluated_scores()
        lower, upper = _adaptive_refinement_interval(ranks, scores)
        refined = allowed_ranks[(allowed_ranks >= lower) & (allowed_ranks <= upper)]
        if np.array_equal(refined, interval):
            evaluated_set = {int(rank) for rank in ranks}
            remaining = np.asarray(
                [rank for rank in interval if int(rank) not in evaluated_set],
                dtype=np.intp,
            )
            evaluate(remaining)
            return
        interval = refined


def _adaptive_tolerance_interval(
    *,
    allowed_ranks: IntArray,
    predictor_ranks: IntArray,
    mean_scores: FloatArray,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> IntArray | None:
    """Return an unresolved failing-to-qualifying adaptive tolerance bracket."""

    selection = _select_tolerant_predictor_rank(
        predictor_ranks,
        mean_scores,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    )
    sorted_ranks = np.sort(predictor_ranks)
    selected_index = int(np.flatnonzero(sorted_ranks == selection.selected_rank)[0])
    if selected_index == 0:
        return None
    lower = int(sorted_ranks[selected_index - 1])
    interval = allowed_ranks[
        (allowed_ranks >= lower) & (allowed_ranks <= selection.selected_rank)
    ]
    if interval.size <= 2:
        return None
    return np.asarray(interval, dtype=np.intp)


def _refine_adaptive_tolerance_boundary(
    *,
    allowed_ranks: IntArray,
    evaluate: Callable[[IntArray], None],
    evaluated_scores: Callable[[], tuple[IntArray, FloatArray]],
    relative_tolerance: float,
    absolute_tolerance: float,
) -> None:
    """Refine adaptive coverage around the public tolerance crossing."""

    while True:
        ranks, scores = evaluated_scores()
        selection = _select_tolerant_predictor_rank(
            ranks,
            scores,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
        )
        interval = _adaptive_tolerance_interval(
            allowed_ranks=allowed_ranks,
            predictor_ranks=ranks,
            mean_scores=scores,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
        )
        if interval is None:
            return

        if interval.size <= _ADAPTIVE_EXHAUSTIVE_THRESHOLD:
            proposed = interval
        else:
            proposed = np.asarray([interval[interval.size // 2]], dtype=np.intp)
        evaluate(proposed)

        updated_ranks, updated_scores = evaluated_scores()
        updated_selection = _select_tolerant_predictor_rank(
            updated_ranks,
            updated_scores,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
        )
        if updated_selection.reference_rank != selection.reference_rank:
            _refine_adaptive_reference(
                allowed_ranks=allowed_ranks,
                evaluate=evaluate,
                evaluated_scores=evaluated_scores,
            )


def _search_adaptive_predictor_ranks(
    *,
    allowed_ranks: IntArray,
    evaluate: Callable[[IntArray], None],
    evaluated_scores: Callable[[], tuple[IntArray, FloatArray]],
    relative_tolerance: float | None = None,
    absolute_tolerance: float | None = None,
) -> None:
    """Evaluate predictor ranks with adaptive exact/tolerance refinement."""

    _refine_adaptive_reference(
        allowed_ranks=allowed_ranks,
        evaluate=evaluate,
        evaluated_scores=evaluated_scores,
    )
    if relative_tolerance is None:
        return
    assert absolute_tolerance is not None
    _refine_adaptive_tolerance_boundary(
        allowed_ranks=allowed_ranks,
        evaluate=evaluate,
        evaluated_scores=evaluated_scores,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    )


def _tied_score_mask(
    scores: ArrayLike,
    reference: float,
    *,
    rtol: float = _NUMERICAL_TIE_RTOL,
    atol: float = _NUMERICAL_TIE_ATOL,
) -> BoolArray:
    """Return scores numerically tied with one finite reference."""

    score_array = np.asarray(scores, dtype=np.float64)
    reference_value = float(reference)
    return np.asarray(
        np.isclose(
            score_array,
            reference_value,
            rtol=rtol,
            atol=atol,
        ),
        dtype=np.bool_,
    )


def _score_tolerance_threshold(
    reference: float,
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> float:
    """Return the simultaneous relative-and-absolute configured-score threshold."""

    with np.errstate(over="ignore", invalid="ignore"):
        relative_threshold = np.float64(reference) - np.float64(
            relative_tolerance
        ) * np.abs(np.float64(reference))
        absolute_threshold = np.float64(reference) - np.float64(absolute_tolerance)
    return float(np.maximum(relative_threshold, absolute_threshold))


def _tolerant_score_mask(
    scores: ArrayLike,
    reference: float,
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> BoolArray:
    """Return scores satisfying both substantive caps around one reference."""

    score_array = np.asarray(scores, dtype=np.float64)
    threshold = _score_tolerance_threshold(
        reference,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    )
    if np.isneginf(threshold):
        return np.ones(score_array.shape, dtype=np.bool_)
    return np.asarray(
        (score_array > threshold) | _tied_score_mask(score_array, threshold),
        dtype=np.bool_,
    )


def _select_tolerant_predictor_rank(
    predictor_ranks: ArrayLike,
    mean_scores: ArrayLike,
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> _PredictorRankScoreSelection:
    """Return exact-reference and smallest tolerance-qualified rank evidence."""

    ranks = np.asarray(predictor_ranks)
    scores = np.asarray(mean_scores, dtype=np.float64)

    reference_score = float(np.max(scores))
    reference_mask = _tied_score_mask(scores, reference_score)
    reference_rank = int(np.min(ranks[reference_mask]))
    selected_mask = _tolerant_score_mask(
        scores,
        reference_score,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    )
    selected_rank = int(np.min(ranks[selected_mask]))
    selected_index = int(np.flatnonzero(ranks == selected_rank)[0])
    threshold = _score_tolerance_threshold(
        reference_score,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    )
    return _PredictorRankScoreSelection(
        reference_rank=reference_rank,
        reference_score=reference_score,
        selected_rank=selected_rank,
        selected_score=float(scores[selected_index]),
        score_threshold=threshold,
    )


def _rank_test_scores(
    mean_scores: ArrayLike,
    *,
    rtol: float = _NUMERICAL_TIE_RTOL,
    atol: float = _NUMERICAL_TIE_ATOL,
) -> IntArray:
    """Return minimum ranks using reference-anchored tolerant score groups."""

    scores = np.asarray(mean_scores, dtype=np.float64)
    order = np.argsort(-scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.intp)
    group_start = 0
    group_reference = float(scores[order[0]])
    for position, index in enumerate(order):
        if position > 0 and not bool(
            _tied_score_mask(
                scores[index],
                group_reference,
                rtol=rtol,
                atol=atol,
            )
        ):
            group_start = position
            group_reference = float(scores[index])
        ranks[index] = group_start + 1
    return ranks


def _as_index_array(index: ArrayLike, *, name: str, n_samples: int) -> IntArray:
    array = np.asarray(index)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {array.shape}.")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if array.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain integer indices.")
    converted = np.asarray(array, dtype=np.intp)
    if np.unique(converted).size != converted.size:
        raise ValueError(f"{name} must not contain duplicate indices.")
    if np.any(converted < 0) or np.any(converted >= n_samples):
        raise ValueError(f"{name} contains an index outside [0, {n_samples}).")
    return converted


def _as_positive_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise ValueError(f"{name} must be a positive finite number; got {value!r}.")
    numeric_value = float(value)
    if not np.isfinite(numeric_value) or numeric_value <= 0.0:
        raise ValueError(f"{name} must be a positive finite number; got {value!r}.")
    return numeric_value
