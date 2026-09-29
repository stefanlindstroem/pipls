from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pipls._core import fit_pipls_core


def _center(array: np.ndarray) -> np.ndarray:
    return array - array.mean(axis=0, keepdims=True)


@pytest.mark.parametrize("response_subspace", ["cross_covariance", "least_squares"])
def test_core_orthogonality_dilation_and_factorization(
    response_subspace: str,
) -> None:
    rng = np.random.default_rng(45)
    X = _center(rng.normal(size=(20, 9)))
    Y = _center(rng.normal(size=(20, 5)))

    result = fit_pipls_core(
        X,
        Y,
        predictor_rank=6,
        n_components=4,
        response_subspace=response_subspace,  # type: ignore[arg-type]
    )

    assert_allclose(result.P.T @ result.P, np.eye(4), atol=1e-12)
    assert_allclose(result.Q.T @ result.Q, np.eye(4), atol=1e-12)
    assert_allclose(result.D, np.diag(np.diag(result.D)), atol=0.0)
    assert np.all(np.diag(result.D) >= 0.0)
    assert np.all(np.diff(np.diag(result.D)) <= 0.0)

    direct_fitted = (X @ result.Pi) @ result.W @ result.C.T
    factored_fitted = X @ result.standardized_regression_map
    assert_allclose(factored_fitted, direct_fitted, atol=2e-12)


def test_core_supports_p_much_greater_than_n() -> None:
    rng = np.random.default_rng(81)
    X = _center(rng.normal(size=(7, 50)))
    Y = _center(rng.normal(size=(7, 3)))

    result = fit_pipls_core(X, Y, predictor_rank=4, n_components=3)

    assert result.x_rank == 6
    assert result.P.shape == (50, 3)
    assert np.all(np.isfinite(result.standardized_regression_map))


def test_least_squares_core_supports_p_much_greater_than_n() -> None:
    rng = np.random.default_rng(20260825)
    X = _center(rng.normal(size=(9, 80)))
    Y = _center(rng.normal(size=(9, 4)))

    result = fit_pipls_core(
        X,
        Y,
        predictor_rank=5,
        n_components=3,
        response_subspace="least_squares",
    )

    assert result.x_rank == 8
    assert result.P.shape == (80, 3)
    assert result.C.shape == (4, 3)
    assert_allclose(result.C.T @ result.C, np.eye(3), atol=1e-12, rtol=1e-12)
    assert np.all(np.isfinite(result.standardized_regression_map))
