"""Shared numerical evaluation for maintained PLS-family path comparisons."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.model_selection import KFold

from pipls import PiPLSRegression, PiPLSSearchCV

from .pls_component_path import PLSComponentPath, evaluate_pls_component_path

REFERENCE_DATASET_NAMES = ("pulp", "sugarcane", "tobacco")
SYNTHETIC_STRESS_CASE = "synthetic_stress"
COMPARISON_CASES = (*REFERENCE_DATASET_NAMES, SYNTHETIC_STRESS_CASE)
ResponseSubspace = Literal["cross_covariance", "least_squares"]
CVSplit = tuple[NDArray[np.intp], NDArray[np.intp]]


@dataclass(frozen=True)
class PLSFamilyPathEvaluation:
    """Numerical result for one matched-CV PLS-family comparison."""

    case: str
    cv_splits: list[CVSplit]
    pipls_searches: dict[ResponseSubspace, PiPLSSearchCV]
    pls_path: PLSComponentPath


def materialize_comparison_cv_splits(
    X: ArrayLike,
    Y: ArrayLike,
) -> list[CVSplit]:
    """Return the maintained seeded five-fold comparison protocol."""

    X_array = np.asarray(X)
    Y_array = np.asarray(Y)
    return [
        (np.asarray(train, dtype=np.intp), np.asarray(validation, dtype=np.intp))
        for train, validation in KFold(
            n_splits=5,
            shuffle=True,
            random_state=0,
        ).split(X_array, Y_array)
    ]


def make_pipls_search(
    case: str,
    *,
    response_subspace: ResponseSubspace,
    cv_splits: list[CVSplit],
) -> PiPLSSearchCV:
    """Build one Pi-PLS search using the maintained case-specific configuration."""

    estimator = PiPLSRegression(
        n_components=1,
        predictor_rank=1,
        response_subspace=response_subspace,
        svd_solver="full" if case == "tobacco" else "auto",
    )
    if case == "pulp":
        return PiPLSSearchCV(estimator=estimator, cv=cv_splits)
    if case == SYNTHETIC_STRESS_CASE:
        return PiPLSSearchCV(
            estimator=estimator,
            search_method="exhaustive",
            cv=cv_splits,
        )
    if case == "sugarcane":
        return PiPLSSearchCV(
            estimator=estimator,
            search_method="adaptive",
            cv=cv_splits,
        )
    if case == "tobacco":
        # Keep Tobacco parallelism bounded because the high-dimensional spectral
        # comparison is expensive and each worker also performs matrix operations.
        return PiPLSSearchCV(
            estimator=estimator,
            search_method="adaptive",
            n_jobs=4,
            cv=cv_splits,
        )
    raise ValueError(f"Unknown comparison case: {case!r}.")


def evaluate_pls_family_paths(
    case: str,
    X: ArrayLike,
    Y: ArrayLike,
    *,
    response_subspaces: Sequence[ResponseSubspace] = (
        "cross_covariance",
        "least_squares",
    ),
) -> PLSFamilyPathEvaluation:
    """Evaluate requested Pi-PLS policies and ordinary PLS on one shared protocol."""

    requested = tuple(response_subspaces)
    if not requested:
        raise ValueError("response_subspaces must contain at least one policy.")
    if len(set(requested)) != len(requested):
        raise ValueError("response_subspaces must not contain duplicate policies.")

    cv_splits = materialize_comparison_cv_splits(X, Y)
    searches: dict[ResponseSubspace, PiPLSSearchCV] = {}
    reference_components: NDArray[np.intp] | None = None
    for response_subspace in requested:
        search = make_pipls_search(
            case,
            response_subspace=response_subspace,
            cv_splits=cv_splits,
        ).fit(X, Y)
        searches[response_subspace] = search
        components = search.component_path_.n_components
        if reference_components is None:
            reference_components = components
        elif not np.array_equal(reference_components, components):
            raise RuntimeError(
                "Requested Pi-PLS response-subspace paths must contain the same component counts."
            )

    if reference_components is None:
        raise RuntimeError("No Pi-PLS component path was evaluated.")

    pls_path = evaluate_pls_component_path(
        X,
        Y,
        max_n_components=int(reference_components[-1]),
        cv=cv_splits,
    )
    if not np.array_equal(reference_components, pls_path.n_components):
        raise RuntimeError("Pi-PLS and ordinary PLS paths must contain the same component counts.")

    return PLSFamilyPathEvaluation(
        case=case,
        cv_splits=cv_splits,
        pipls_searches=searches,
        pls_path=pls_path,
    )
