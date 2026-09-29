from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import KFold

from pipls._model_selection import (
    _adaptive_refinement_interval,
    _epv_predictor_rank,
    _hard_predictor_rank_limit,
    _logarithmic_predictor_rank_values,
    _materialize_cv_splits,
    _rank_test_scores,
    _score_tolerance_threshold,
    _search_adaptive_predictor_ranks,
    _select_tolerant_predictor_rank,
    _tied_score_mask,
    _tolerant_score_mask,
)


def test_hard_predictor_rank_limit_uses_centered_training_fold_cap() -> None:
    assert (
        _hard_predictor_rank_limit(
            n_features=100,
            n_train_min=51,
        )
        == 50
    )


def test_hard_predictor_rank_limit_respects_feature_cap() -> None:
    assert (
        _hard_predictor_rank_limit(
            n_features=3,
            n_train_min=100,
        )
        == 3
    )


def test_hard_predictor_rank_limit_respects_smallest_training_fold() -> None:
    complete_data_bound = _hard_predictor_rank_limit(
        n_features=30,
        n_train_min=50,
    )
    fold_bound = _hard_predictor_rank_limit(
        n_features=30,
        n_train_min=39,
    )
    feasibility_capped = _hard_predictor_rank_limit(
        n_features=30,
        n_train_min=4,
    )

    assert complete_data_bound == 30
    assert fold_bound == 30
    assert feasibility_capped == 3


@pytest.mark.parametrize(
    ("samples_per_predictor_rank", "expected"),
    [
        (10.0, 6),
        (5.0, 11),
        (1.0, 51),
        (7.5, 7),
    ],
)
def test_epv_predictor_rank_uses_full_sample_ceiling_rule(
    samples_per_predictor_rank: float,
    expected: int,
) -> None:
    assert (
        _epv_predictor_rank(
            n_features=100,
            n_samples=51,
            samples_per_predictor_rank=samples_per_predictor_rank,
        )
        == expected
    )


def test_epv_predictor_rank_respects_feature_cap() -> None:
    assert (
        _epv_predictor_rank(
            n_features=3,
            n_samples=100,
            samples_per_predictor_rank=10,
        )
        == 3
    )


def test_epv_predictor_rank_handles_tiny_positive_support_parameter() -> None:
    assert (
        _epv_predictor_rank(
            n_features=100,
            n_samples=51,
            samples_per_predictor_rank=np.nextafter(0.0, 1.0),
        )
        == 100
    )


def test_materialize_cv_splits_reuses_one_concrete_split_set() -> None:
    X = np.arange(30, dtype=np.float64).reshape(10, 3)
    y = np.arange(10, dtype=np.float64)

    materialized = _materialize_cv_splits(KFold(n_splits=4), X, y)

    assert len(materialized.splits) == 4
    assert materialized.n_train_min == 7
    np.testing.assert_array_equal(materialized.splits[0][1], np.array([0, 1, 2]))
    assert all(train.dtype == np.intp for train, _ in materialized.splits)
    assert all(validation.dtype == np.intp for _, validation in materialized.splits)
    assert all(not train.flags.writeable for train, _ in materialized.splits)
    assert all(not validation.flags.writeable for _, validation in materialized.splits)


def test_materialize_cv_splits_accepts_an_iterable_and_copies_indices() -> None:
    X = np.zeros((5, 2))
    y = np.zeros((5, 1))
    train = np.array([0, 1, 2], dtype=np.int64)
    validation = np.array([3, 4], dtype=np.int64)

    materialized = _materialize_cv_splits([(train, validation)], X, y)
    train[0] = 4
    validation[0] = 0

    np.testing.assert_array_equal(materialized.splits[0][0], np.array([0, 1, 2]))
    np.testing.assert_array_equal(materialized.splits[0][1], np.array([3, 4]))
    assert materialized.n_train_min == 3


@pytest.mark.parametrize(
    "splits",
    [
        [],
        [([0, 1], [])],
        [([0, 1], [1, 2])],
        [([0, 0], [1, 2])],
        [([0, 1], [2, 5])],
    ],
)
def test_materialize_cv_splits_rejects_invalid_splits(splits: object) -> None:
    X = np.zeros((5, 2))
    y = np.zeros(5)

    with pytest.raises(ValueError):
        _materialize_cv_splits(splits, X, y)


def test_logarithmic_predictor_rank_values_are_deterministic_and_include_endpoints() -> None:
    values = _logarithmic_predictor_rank_values(lower=2, upper=100)

    np.testing.assert_array_equal(values, np.array([2, 4, 7, 14, 27, 52, 100]))
    assert values[0] == 2
    assert values[-1] == 100
    assert np.all(np.diff(values) > 0)


def test_logarithmic_predictor_rank_values_handle_small_intervals() -> None:
    np.testing.assert_array_equal(
        _logarithmic_predictor_rank_values(lower=4, upper=4),
        np.array([4]),
    )


def test_adaptive_refinement_interval_uses_neighbors_around_best_rank() -> None:
    interval = _adaptive_refinement_interval(
        np.array([2, 4, 8, 16, 32]),
        np.array([-5.0, -3.0, -1.0, -2.0, -4.0]),
    )

    assert interval == (4, 16)


def test_adaptive_refinement_interval_respects_lower_rank_ties() -> None:
    interval = _adaptive_refinement_interval(
        np.array([2, 4, 8, 16]),
        np.array([3.0, 3.0, 2.0, 1.0]),
    )

    assert interval == (2, 4)


def _run_adaptive_rank_surface(
    score_by_rank: dict[int, float],
    *,
    relative_tolerance: float | None,
    absolute_tolerance: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    allowed = np.asarray(sorted(score_by_rank), dtype=np.intp)
    evaluated: dict[int, float] = {}

    def evaluate(ranks: np.ndarray) -> None:
        for rank in ranks:
            evaluated.setdefault(int(rank), score_by_rank[int(rank)])

    def evaluated_scores() -> tuple[np.ndarray, np.ndarray]:
        ranks = np.asarray(sorted(evaluated), dtype=np.intp)
        scores = np.asarray([evaluated[int(rank)] for rank in ranks], dtype=np.float64)
        return ranks, scores

    _search_adaptive_predictor_ranks(
        allowed_ranks=allowed,
        evaluate=evaluate,
        evaluated_scores=evaluated_scores,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    )
    return evaluated_scores()


def test_adaptive_rank_search_refines_tolerance_boundary() -> None:
    score_by_rank = {rank: rank / 50.0 for rank in range(1, 51)}

    ranks, scores = _run_adaptive_rank_surface(
        score_by_rank,
        relative_tolerance=0.50,
        absolute_tolerance=np.inf,
    )
    selection = _select_tolerant_predictor_rank(
        ranks,
        scores,
        relative_tolerance=0.50,
        absolute_tolerance=np.inf,
    )

    assert selection.reference_rank == 50
    assert selection.selected_rank == 25
    assert 24 in ranks
    assert 25 in ranks
    assert 15 not in ranks


def test_zero_tolerance_does_not_extend_exact_reference_coverage() -> None:
    score_by_rank = {rank: rank / 50.0 for rank in range(1, 51)}

    exact_ranks, _ = _run_adaptive_rank_surface(
        score_by_rank,
        relative_tolerance=None,
        absolute_tolerance=None,
    )
    tolerant_ranks, _ = _run_adaptive_rank_surface(
        score_by_rank,
        relative_tolerance=0.0,
        absolute_tolerance=0.0,
    )

    np.testing.assert_array_equal(tolerant_ranks, exact_ranks)


def test_tolerance_refinement_rechecks_a_changed_exact_reference() -> None:
    score_by_rank = {rank: rank / 50.0 for rank in range(1, 51)}
    score_by_rank[20] = 1.20

    ranks, scores = _run_adaptive_rank_surface(
        score_by_rank,
        relative_tolerance=0.50,
        absolute_tolerance=np.inf,
    )
    selection = _select_tolerant_predictor_rank(
        ranks,
        scores,
        relative_tolerance=0.50,
        absolute_tolerance=np.inf,
    )

    assert selection.reference_rank == 20
    assert selection.selected_rank == 20
    assert 19 in ranks
    assert 21 in ranks


def test_rank_test_scores_assigns_minimum_rank_to_ties() -> None:
    ranks = _rank_test_scores(np.array([0.5, 0.5, 0.2, 0.1, 0.2]))

    np.testing.assert_array_equal(ranks, np.array([1, 1, 3, 5, 3]))


def test_rank_test_scores_do_not_chain_adjacent_tolerant_comparisons() -> None:
    ranks = _rank_test_scores(
        np.array([10.0, 9.4, 8.8, 8.2]),
        rtol=0.0,
        atol=0.75,
    )

    np.testing.assert_array_equal(ranks, np.array([1, 1, 3, 3]))


def test_tied_score_mask_uses_one_reference_score() -> None:
    tied = _tied_score_mask(
        np.array([10.0, 9.4, 8.8]),
        10.0,
        rtol=0.0,
        atol=0.75,
    )

    np.testing.assert_array_equal(tied, np.array([True, True, False]))


@pytest.mark.parametrize(
    ("reference", "relative_tolerance", "absolute_tolerance", "expected"),
    [
        (10.0, 0.10, np.inf, 9.0),
        (-10.0, 0.10, np.inf, -11.0),
        (0.0, 0.10, np.inf, 0.0),
        (10.0, 1.0, 0.25, 9.75),
        (-10.0, 1.0, 0.25, -10.25),
        (10.0, 0.10, 0.25, 9.75),
    ],
)
def test_score_tolerance_threshold_uses_simultaneous_caps(
    reference: float,
    relative_tolerance: float,
    absolute_tolerance: float,
    expected: float,
) -> None:
    threshold = _score_tolerance_threshold(
        reference,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    )

    assert threshold == pytest.approx(expected)


def test_score_tolerance_threshold_allows_an_effectively_unbounded_allowance() -> None:
    threshold = _score_tolerance_threshold(
        np.finfo(np.float64).max,
        relative_tolerance=np.finfo(np.float64).max,
        absolute_tolerance=np.inf,
    )

    assert np.isneginf(threshold)


def test_tolerant_score_mask_includes_the_exact_boundary() -> None:
    mask = _tolerant_score_mask(
        np.array([10.0, 9.75, 9.749, 9.0]),
        10.0,
        relative_tolerance=0.10,
        absolute_tolerance=0.25,
    )

    np.testing.assert_array_equal(mask, np.array([True, True, False, False]))


def test_tolerant_score_mask_keeps_numerical_equality_separate_from_allowance() -> None:
    mask = _tolerant_score_mask(
        np.array([1.0, 1.0 - 5e-14, 1.0 - 1e-8]),
        1.0,
        relative_tolerance=0.0,
        absolute_tolerance=0.0,
    )

    np.testing.assert_array_equal(mask, np.array([True, True, False]))


def test_select_tolerant_predictor_rank_returns_reference_and_selected_evidence() -> None:
    selection = _select_tolerant_predictor_rank(
        np.array([8, 2, 6, 4], dtype=np.intp),
        np.array([-1.00, -1.08, -1.04, -1.06]),
        relative_tolerance=0.10,
        absolute_tolerance=np.inf,
    )

    assert selection.reference_rank == 8
    assert selection.reference_score == pytest.approx(-1.00)
    assert selection.selected_rank == 2
    assert selection.selected_score == pytest.approx(-1.08)
    assert selection.score_threshold == pytest.approx(-1.10)


def test_select_tolerant_predictor_rank_applies_both_caps() -> None:
    selection = _select_tolerant_predictor_rank(
        np.array([2, 4, 6], dtype=np.intp),
        np.array([9.6, 9.8, 10.0]),
        relative_tolerance=0.10,
        absolute_tolerance=0.25,
    )

    assert selection.reference_rank == 6
    assert selection.selected_rank == 4
    assert selection.score_threshold == pytest.approx(9.75)


def test_select_tolerant_predictor_rank_uses_low_rank_numerical_reference_ties() -> None:
    selection = _select_tolerant_predictor_rank(
        np.array([5, 2, 4], dtype=np.intp),
        np.array([0.7, 0.8 - 5e-14, 0.8]),
        relative_tolerance=0.0,
        absolute_tolerance=0.0,
    )

    assert selection.reference_rank == 2
    assert selection.reference_score == pytest.approx(0.8)
    assert selection.selected_rank == 2
    assert selection.selected_score == pytest.approx(0.8 - 5e-14)


def test_adaptive_refinement_remains_anchored_to_the_exact_optimum() -> None:
    ranks = np.array([2, 4, 8, 16, 32], dtype=np.intp)
    scores = -np.array([1.09, 1.08, 1.07, 1.00, 1.05])
    tolerant = _select_tolerant_predictor_rank(
        ranks,
        scores,
        relative_tolerance=0.10,
        absolute_tolerance=np.inf,
    )

    assert tolerant.selected_rank == 2
    assert _adaptive_refinement_interval(ranks, scores) == (8, 32)
