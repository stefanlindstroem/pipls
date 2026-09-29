"""Immutable concise results for a fitted Π-PLS component path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ._model_selection import (
    _score_tolerance_threshold,
    _tied_score_mask,
    _tolerant_score_mask,
)
from ._result_validation import _read_only_float_array, _read_only_int_array

__all__ = [
    "PiPLSComponentPath",
    "PiPLSPredictorRankEvidence",
    "PiPLSPredictorRankProfile",
    "PiPLSSelection",
]

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.intp]
SelectionRule = Literal["best_score", "minimum_cv_mse"]
PredictorRankPolicy = Literal["optimized", "fixed", "epv"]


def _cv_mse_tolerance_threshold(
    minimum_cv_mse: float,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> float:
    """Return the simultaneous relative-and-absolute CV-MSE threshold."""

    with np.errstate(over="ignore", invalid="ignore"):
        relative_threshold = np.float64(minimum_cv_mse) * np.float64(
            1.0 + relative_tolerance
        )
        absolute_threshold = np.float64(minimum_cv_mse) + np.float64(
            absolute_tolerance
        )
    return float(np.minimum(relative_threshold, absolute_threshold))


@dataclass(frozen=True)
class PiPLSPredictorRankEvidence:
    """Immutable provenance for one conditional predictor-rank choice.

    Attributes
    ----------
    reference_predictor_rank : int
        Smallest predictor rank numerically tied at the exact configured-score
        optimum.
    reference_mean_test_score : float
        Exact reference optimum in configured-score units.
    reference_cv_mse_mean : float
        Mean response-standardized validation MSE for the reference candidate.
    reference_cv_mse_std : float
        Population standard deviation of response-standardized validation MSE
        across splits for the reference candidate.
    relative_tolerance : float
        Resolved finite nonnegative configured-score relative tolerance.
    absolute_tolerance : float
        Resolved nonnegative configured-score absolute tolerance; positive
        infinity disables the absolute cap.
    score_threshold : float
        Derived effective configured-score threshold.
    """

    reference_predictor_rank: int
    reference_mean_test_score: float
    reference_cv_mse_mean: float
    reference_cv_mse_std: float
    relative_tolerance: float
    absolute_tolerance: float

    @property
    def score_threshold(self) -> float:
        """Return the effective configured-score threshold."""

        return _score_tolerance_threshold(
            self.reference_mean_test_score,
            relative_tolerance=self.relative_tolerance,
            absolute_tolerance=self.absolute_tolerance,
        )


@dataclass(frozen=True)
class PiPLSSelection:
    """Immutable Π-PLS selection for one evaluated pair of component count and predictor rank.

    Attributes
    ----------
    n_components : int
        Evaluated number of paired latent modes.
    predictor_rank : int
        Predictor rank selected conditionally for ``n_components``.
    predictor_rank_policy : {"optimized", "fixed", "epv"}
        Interpretation of the predictor-rank specification used by the path.
    mean_test_score : float
        Mean configured test score for the selected candidate.
    cv_mse_mean : float
        Mean response-standardized validation MSE.
    cv_mse_std : float
        Population standard deviation of response-standardized MSE across
        validation splits.
    n_splits : int
        Number of cross-validation splits.
    rule : {"best_score", "minimum_cv_mse"} or None
        Search-owned rule that produced this selection. ``None`` denotes direct
        lookup by component count.
    reference_minimum : PiPLSSelection or None
        Exact minimum-CV-MSE path row used to derive a tolerance selection.
        The reference row has no rule provenance.
    relative_tolerance : float or None
        Resolved nonnegative relative tolerance for ``rule="minimum_cv_mse"``.
    absolute_tolerance : float or None
        Resolved nonnegative absolute tolerance for ``rule="minimum_cv_mse"``;
        positive infinity disables the absolute cap.
    cv_mse_threshold : float or None
        Derived effective threshold for ``rule="minimum_cv_mse"``.
    predictor_rank_evidence : PiPLSPredictorRankEvidence or None
        Conditional predictor-rank reference and resolved tolerance provenance.
    """

    n_components: int
    predictor_rank: int
    predictor_rank_policy: PredictorRankPolicy
    mean_test_score: float
    cv_mse_mean: float
    cv_mse_std: float
    n_splits: int
    rule: SelectionRule | None = None
    reference_minimum: PiPLSSelection | None = None
    relative_tolerance: float | None = None
    absolute_tolerance: float | None = None
    predictor_rank_evidence: PiPLSPredictorRankEvidence | None = None

    @property
    def cv_mse_threshold(self) -> float | None:
        """Return the effective threshold for minimum-CV-MSE selection."""

        if self.rule != "minimum_cv_mse":
            return None
        reference = self.reference_minimum
        relative_tolerance = self.relative_tolerance
        absolute_tolerance = self.absolute_tolerance
        if (
            reference is None
            or relative_tolerance is None
            or absolute_tolerance is None
        ):  # pragma: no cover - producer-owned invariant
            raise RuntimeError(
                "A minimum-CV-MSE selection requires complete tolerance provenance."
            )
        return _cv_mse_tolerance_threshold(
            reference.cv_mse_mean,
            relative_tolerance,
            absolute_tolerance,
        )


@dataclass(frozen=True)
class PiPLSPredictorRankProfile:
    """Immutable evaluated predictor-rank results for one paired-mode count.

    Arrays contain the predictor-rank candidates actually evaluated for one
    paired-mode count, sorted by ascending predictor rank. ``reference_selection``
    exposes the exact configured-score optimum, while ``selection`` exposes the
    smallest evaluated rank satisfying the fitted predictor-rank tolerances.

    Attributes
    ----------
    n_components : int
        Number of paired latent modes shared by every evaluated candidate.
    predictor_rank : ndarray of shape (n_evaluated_ranks,)
        Evaluated predictor ranks in strictly ascending order.
    mean_test_score : ndarray of shape (n_evaluated_ranks,)
        Mean configured test score for each evaluated rank.
    cv_mse_mean : ndarray of shape (n_evaluated_ranks,)
        Mean response-standardized validation MSE for each evaluated rank.
    cv_mse_std : ndarray of shape (n_evaluated_ranks,)
        Population standard deviation of response-standardized MSE across
        validation splits.
    predictor_rank_policy : {"optimized", "fixed", "epv"}
        Predictor-rank policy shared by every evaluated candidate.
    n_splits : int
        Number of cross-validation splits.
    reference_selection : PiPLSSelection
        Exact configured-score reference optimum without tolerance provenance.
    selection : PiPLSSelection
        Smallest evaluated tolerance-qualified conditional selection.
    predictor_rank_evidence : PiPLSPredictorRankEvidence or None
        Predictor-rank tolerance provenance for optimized policies.
    """

    n_components: int
    predictor_rank: IntArray
    mean_test_score: FloatArray
    cv_mse_mean: FloatArray
    cv_mse_std: FloatArray
    predictor_rank_policy: PredictorRankPolicy
    n_splits: int
    predictor_rank_evidence: PiPLSPredictorRankEvidence | None = None

    def __post_init__(self) -> None:
        """Store defensive read-only copies of aligned numerical arrays."""

        object.__setattr__(
            self,
            "predictor_rank",
            _read_only_int_array(self.predictor_rank, name="predictor_rank"),
        )
        object.__setattr__(
            self,
            "mean_test_score",
            _read_only_float_array(self.mean_test_score, name="mean_test_score"),
        )
        object.__setattr__(
            self,
            "cv_mse_mean",
            _read_only_float_array(self.cv_mse_mean, name="cv_mse_mean"),
        )
        object.__setattr__(
            self,
            "cv_mse_std",
            _read_only_float_array(self.cv_mse_std, name="cv_mse_std"),
        )

    @property
    def reference_selection(self) -> PiPLSSelection:
        """Return the exact configured-score reference optimum."""

        maximum = float(np.max(self.mean_test_score))
        tied = np.flatnonzero(_tied_score_mask(self.mean_test_score, maximum))
        index = int(tied[0])
        return PiPLSSelection(
            n_components=self.n_components,
            predictor_rank=int(self.predictor_rank[index]),
            predictor_rank_policy=self.predictor_rank_policy,
            mean_test_score=float(self.mean_test_score[index]),
            cv_mse_mean=float(self.cv_mse_mean[index]),
            cv_mse_std=float(self.cv_mse_std[index]),
            n_splits=self.n_splits,
        )

    @property
    def selection(self) -> PiPLSSelection:
        """Return the conditional predictor-rank selection."""

        evidence = self.predictor_rank_evidence
        if evidence is None:
            return self.reference_selection
        qualifying = np.flatnonzero(
            _tolerant_score_mask(
                self.mean_test_score,
                evidence.reference_mean_test_score,
                relative_tolerance=evidence.relative_tolerance,
                absolute_tolerance=evidence.absolute_tolerance,
            )
        )
        index = int(qualifying[0])
        return PiPLSSelection(
            n_components=self.n_components,
            predictor_rank=int(self.predictor_rank[index]),
            predictor_rank_policy=self.predictor_rank_policy,
            mean_test_score=float(self.mean_test_score[index]),
            cv_mse_mean=float(self.cv_mse_mean[index]),
            cv_mse_std=float(self.cv_mse_std[index]),
            n_splits=self.n_splits,
            predictor_rank_evidence=evidence,
        )

    def __reduce__(self) -> tuple[type[PiPLSPredictorRankProfile], tuple[object, ...]]:
        """Reconstruct so unpickled arrays remain read-only."""

        return (
            type(self),
            (
                self.n_components,
                self.predictor_rank,
                self.mean_test_score,
                self.cv_mse_mean,
                self.cv_mse_std,
                self.predictor_rank_policy,
                self.n_splits,
                self.predictor_rank_evidence,
            ),
        )


@dataclass(frozen=True)
class PiPLSComponentPath:
    """Immutable concise component-path results.

    Each array contains one conditionally selected predictor-rank result per
    evaluated paired-mode count. Arrays are defensive, read-only copies and are
    aligned by row.

    Attributes
    ----------
    n_components : ndarray of shape (n_component_values,)
        Evaluated paired-mode counts in strictly ascending order.
    predictor_rank : ndarray of shape (n_component_values,)
        Conditionally selected predictor rank for each paired-mode count.
    predictor_rank_policy : {"optimized", "fixed", "epv"}
        Predictor-rank policy shared by every path row.
    mean_test_score : ndarray of shape (n_component_values,)
        Mean configured test score for each selected candidate.
    cv_mse_mean : ndarray of shape (n_component_values,)
        Mean response-standardized validation MSE for each selected candidate.
    cv_mse_std : ndarray of shape (n_component_values,)
        Population standard deviation of response-standardized MSE across
        validation splits.
    n_splits : int
        Number of cross-validation splits shared by every path row.
    predictor_rank_evidence : tuple of PiPLSPredictorRankEvidence or None
        Row-aligned predictor-rank tolerance provenance for optimized policies.
    """

    n_components: IntArray
    predictor_rank: IntArray
    predictor_rank_policy: PredictorRankPolicy
    mean_test_score: FloatArray
    cv_mse_mean: FloatArray
    cv_mse_std: FloatArray
    n_splits: int
    predictor_rank_evidence: tuple[PiPLSPredictorRankEvidence, ...] | None = None

    def __post_init__(self) -> None:
        """Store defensive read-only copies of path arrays and evidence."""

        object.__setattr__(
            self,
            "n_components",
            _read_only_int_array(self.n_components, name="n_components"),
        )
        object.__setattr__(
            self,
            "predictor_rank",
            _read_only_int_array(self.predictor_rank, name="predictor_rank"),
        )
        object.__setattr__(
            self,
            "mean_test_score",
            _read_only_float_array(self.mean_test_score, name="mean_test_score"),
        )
        object.__setattr__(
            self,
            "cv_mse_mean",
            _read_only_float_array(self.cv_mse_mean, name="cv_mse_mean"),
        )
        object.__setattr__(
            self,
            "cv_mse_std",
            _read_only_float_array(self.cv_mse_std, name="cv_mse_std"),
        )
        evidence = self.predictor_rank_evidence
        if evidence is not None:
            object.__setattr__(self, "predictor_rank_evidence", tuple(evidence))

    def __reduce__(self) -> tuple[type[PiPLSComponentPath], tuple[object, ...]]:
        """Reconstruct so unpickled arrays remain read-only."""

        return (
            type(self),
            (
                self.n_components,
                self.predictor_rank,
                self.predictor_rank_policy,
                self.mean_test_score,
                self.cv_mse_mean,
                self.cv_mse_std,
                self.n_splits,
                self.predictor_rank_evidence,
            ),
        )

    def _selection_at_index(self, index: int) -> PiPLSSelection:
        """Return one selection from an internally validated row index."""

        return PiPLSSelection(
            n_components=int(self.n_components[index]),
            predictor_rank=int(self.predictor_rank[index]),
            predictor_rank_policy=self.predictor_rank_policy,
            mean_test_score=float(self.mean_test_score[index]),
            cv_mse_mean=float(self.cv_mse_mean[index]),
            cv_mse_std=float(self.cv_mse_std[index]),
            n_splits=self.n_splits,
            predictor_rank_evidence=(
                None
                if self.predictor_rank_evidence is None
                else self.predictor_rank_evidence[index]
            ),
        )
