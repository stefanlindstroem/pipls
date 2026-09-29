from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_support_module() -> ModuleType:
    examples_dir = _repository_root() / "examples"
    sys.path.insert(0, str(examples_dir))
    try:
        return importlib.import_module("_support.pls_family_path_comparison")
    finally:
        sys.path.remove(str(examples_dir))


PLS_FAMILY_PATHS = _load_support_module()


def test_family_path_evaluator_reuses_one_materialized_protocol() -> None:
    rng = np.random.default_rng(1729)
    X = rng.normal(size=(30, 6))
    Y = rng.normal(size=(30, 3))

    evaluation = PLS_FAMILY_PATHS.evaluate_pls_family_paths(
        "pulp",
        X,
        Y,
        response_subspaces=("cross_covariance", "least_squares"),
    )

    cross_search = evaluation.pipls_searches["cross_covariance"]
    least_squares_search = evaluation.pipls_searches["least_squares"]
    assert cross_search.cv is evaluation.cv_splits
    assert least_squares_search.cv is evaluation.cv_splits
    assert cross_search.n_splits_ == least_squares_search.n_splits_ == 5
    assert evaluation.pls_path.n_splits == 5

    cross_components = cross_search.component_path_.n_components
    least_squares_components = least_squares_search.component_path_.n_components
    np.testing.assert_array_equal(cross_components, least_squares_components)
    np.testing.assert_array_equal(cross_components, evaluation.pls_path.n_components)
    assert np.isfinite(cross_search.component_path_.cv_mse_mean).all()
    assert np.isfinite(least_squares_search.component_path_.cv_mse_mean).all()
    assert np.isfinite(evaluation.pls_path.cv_mse_mean).all()


def test_family_path_evaluator_validates_requested_response_policies() -> None:
    X = np.arange(30, dtype=float).reshape(10, 3)
    Y = np.arange(20, dtype=float).reshape(10, 2)

    with pytest.raises(ValueError, match="at least one"):
        PLS_FAMILY_PATHS.evaluate_pls_family_paths(
            "pulp",
            X,
            Y,
            response_subspaces=(),
        )
    with pytest.raises(ValueError, match="duplicate"):
        PLS_FAMILY_PATHS.evaluate_pls_family_paths(
            "pulp",
            X,
            Y,
            response_subspaces=("cross_covariance", "cross_covariance"),
        )
