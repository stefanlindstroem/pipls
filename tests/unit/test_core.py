from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pipls._core import fit_pipls_core


def _center(array: np.ndarray) -> np.ndarray:
    return array - array.mean(axis=0, keepdims=True)


def test_cross_covariance_response_basis_matches_direct_svd() -> None:
    from pipls._core import _cross_covariance_response_basis

    rng = np.random.default_rng(20260819)
    Z = _center(rng.normal(size=(18, 6)))
    Y = _center(rng.normal(size=(18, 4)))

    cross_product = Z.T @ Y
    _, _, cross_vt = np.linalg.svd(cross_product, full_matrices=False)
    expected = np.asarray(cross_vt[:3, :].T, dtype=np.float64)

    actual = _cross_covariance_response_basis(
        Z,
        Y,
        n_components=3,
    )

    np.testing.assert_array_equal(actual, expected)


def test_least_squares_response_basis_matches_choice_c_subspace() -> None:
    from pipls._core import _least_squares_response_basis

    rng = np.random.default_rng(20260820)
    Z = _center(rng.normal(size=(24, 6)))
    Y = _center(rng.normal(size=(24, 5)))

    actual = _least_squares_response_basis(
        Z,
        Y,
        n_components=3,
    )

    cross_product = Z.T @ Y
    gram = Z.T @ Z
    choice_c = cross_product.T @ np.linalg.solve(gram, cross_product)
    choice_c = 0.5 * (choice_c + choice_c.T)
    _, eigenvectors = np.linalg.eigh(choice_c)
    expected = eigenvectors[:, -3:]

    assert actual.shape == (5, 3)
    assert np.all(np.isfinite(actual))
    assert_allclose(actual.T @ actual, np.eye(3), atol=1e-12, rtol=1e-12)
    assert_allclose(
        actual @ actual.T,
        expected @ expected.T,
        atol=1e-12,
        rtol=1e-12,
    )


def test_core_response_subspace_default_matches_explicit_cross_covariance() -> None:
    rng = np.random.default_rng(20260821)
    X = _center(rng.normal(size=(22, 8)))
    Y = _center(rng.normal(size=(22, 5)))

    default = fit_pipls_core(
        X,
        Y,
        predictor_rank=6,
        n_components=3,
    )
    explicit = fit_pipls_core(
        X,
        Y,
        predictor_rank=6,
        n_components=3,
        response_subspace="cross_covariance",
    )

    for name in ("Pi", "C", "W", "P", "D", "Q"):
        np.testing.assert_array_equal(
            getattr(default, name),
            getattr(explicit, name),
        )
    np.testing.assert_array_equal(
        default.standardized_regression_map,
        explicit.standardized_regression_map,
    )


def test_core_least_squares_response_subspace_completes_factorization() -> None:
    rng = np.random.default_rng(20260822)
    X = _center(rng.normal(size=(28, 9)))
    Y = _center(rng.normal(size=(28, 5)))

    result = fit_pipls_core(
        X,
        Y,
        predictor_rank=6,
        n_components=3,
        response_subspace="least_squares",
    )

    assert result.C.shape == (5, 3)
    assert_allclose(result.C.T @ result.C, np.eye(3), atol=1e-12, rtol=1e-12)
    direct_fitted = (X @ result.Pi) @ result.W @ result.C.T
    factored_fitted = X @ result.standardized_regression_map
    assert_allclose(factored_fitted, direct_fitted, atol=2e-12, rtol=1e-12)


def test_least_squares_response_subspace_matches_reduced_rank_regression_fit() -> None:
    rng = np.random.default_rng(20260826)
    X = _center(rng.normal(size=(30, 9)))
    Y = _center(rng.normal(size=(30, 5)))
    h = 3

    result = fit_pipls_core(
        X,
        Y,
        predictor_rank=6,
        n_components=h,
        response_subspace="least_squares",
    )

    Z = X @ result.Pi
    unrestricted_coefficients = np.linalg.lstsq(Z, Y, rcond=None)[0]
    unrestricted_fitted = Z @ unrestricted_coefficients
    left, singular_values, right_t = np.linalg.svd(
        unrestricted_fitted,
        full_matrices=False,
    )
    expected_rrr_fitted = (
        left[:, :h] * singular_values[:h]
    ) @ right_t[:h, :]
    actual_fitted = X @ result.standardized_regression_map

    assert_allclose(actual_fitted, expected_rrr_fitted, atol=3e-12, rtol=2e-12)


def test_least_squares_response_subspace_minimizes_training_residual() -> None:
    rng = np.random.default_rng(20260715)
    X = _center(rng.normal(size=(8, 5)))
    Y = _center(rng.normal(size=(8, 3)))

    covariance = fit_pipls_core(
        X,
        Y,
        predictor_rank=3,
        n_components=1,
        response_subspace="cross_covariance",
    )
    least_squares = fit_pipls_core(
        X,
        Y,
        predictor_rank=3,
        n_components=1,
        response_subspace="least_squares",
    )

    covariance_error = np.linalg.norm(
        Y - X @ covariance.standardized_regression_map,
        ord="fro",
    ) ** 2
    least_squares_error = np.linalg.norm(
        Y - X @ least_squares.standardized_regression_map,
        ord="fro",
    ) ** 2

    assert least_squares_error <= covariance_error + 1e-12
    assert covariance_error - least_squares_error > 1e-8


def test_response_subspace_policies_are_equivalent_for_one_response() -> None:
    rng = np.random.default_rng(20260827)
    X = _center(rng.normal(size=(24, 8)))
    Y = _center(rng.normal(size=(24, 1)))

    covariance = fit_pipls_core(
        X,
        Y,
        predictor_rank=5,
        n_components=1,
        response_subspace="cross_covariance",
    )
    least_squares = fit_pipls_core(
        X,
        Y,
        predictor_rank=5,
        n_components=1,
        response_subspace="least_squares",
    )

    assert_allclose(
        least_squares.standardized_regression_map,
        covariance.standardized_regression_map,
        atol=2e-12,
        rtol=1e-12,
    )


def test_response_subspace_policies_match_with_full_response_subspace() -> None:
    rng = np.random.default_rng(20260828)
    X = _center(rng.normal(size=(25, 7)))
    Y = _center(rng.normal(size=(25, 3)))

    covariance = fit_pipls_core(
        X,
        Y,
        predictor_rank=5,
        n_components=3,
        response_subspace="cross_covariance",
    )
    least_squares = fit_pipls_core(
        X,
        Y,
        predictor_rank=5,
        n_components=3,
        response_subspace="least_squares",
    )
    Z = X @ covariance.Pi
    expected_map = covariance.Pi @ np.linalg.lstsq(Z, Y, rcond=None)[0]

    assert_allclose(
        covariance.standardized_regression_map,
        expected_map,
        atol=3e-12,
        rtol=2e-12,
    )
    assert_allclose(
        least_squares.standardized_regression_map,
        expected_map,
        atol=3e-12,
        rtol=2e-12,
    )


def test_core_rejects_invalid_response_subspace() -> None:
    rng = np.random.default_rng(20260823)
    X = _center(rng.normal(size=(12, 5)))
    Y = _center(rng.normal(size=(12, 3)))

    with pytest.raises(
        ValueError,
        match='response_subspace must be "cross_covariance" or "least_squares"',
    ):
        fit_pipls_core(
            X,
            Y,
            predictor_rank=4,
            n_components=2,
            response_subspace="lstsq",  # type: ignore[arg-type]
        )

def test_core_shapes_and_regression_map() -> None:
    rng = np.random.default_rng(12)
    X = _center(rng.normal(size=(15, 7)))
    Y = _center(rng.normal(size=(15, 4)))

    result = fit_pipls_core(X, Y, predictor_rank=5, n_components=3)

    assert result.Pi.shape == (7, 5)
    assert result.C.shape == (4, 3)
    assert result.W.shape == (5, 3)
    assert result.P.shape == (7, 3)
    assert result.D.shape == (3, 3)
    assert result.Q.shape == (4, 3)
    assert result.standardized_regression_map.shape == (7, 4)
    assert_allclose(
        result.standardized_regression_map,
        result.P @ result.D @ result.Q.T,
    )
    prediction = X @ result.standardized_regression_map
    assert prediction.shape == Y.shape
    assert np.all(np.isfinite(prediction))


def test_core_rejects_inadmissible_inputs() -> None:
    X = np.eye(4)
    Y = np.ones((4, 2))

    with pytest.raises(ValueError, match="same number of samples"):
        fit_pipls_core(X, Y[:3], predictor_rank=2, n_components=1)
    with pytest.raises(ValueError, match="positive integer"):
        fit_pipls_core(X, Y, predictor_rank=1.5, n_components=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"min\(predictor_rank, n_targets\)"):
        fit_pipls_core(X, Y, predictor_rank=2, n_components=3)
    with pytest.raises(ValueError, match="finite"):
        fit_pipls_core(np.array([[np.nan]]), np.ones((1, 1)), predictor_rank=1, n_components=1)


def test_core_handles_rank_deficiency_explicitly() -> None:
    base = np.arange(1.0, 9.0).reshape(4, 2)
    X = np.column_stack([base, base[:, 0] + base[:, 1]])
    Y = np.column_stack([base[:, 0], base[:, 1]])

    result = fit_pipls_core(X, Y, predictor_rank=2, n_components=2)
    assert result.x_rank == 2

    with pytest.raises(ValueError, match="numerical rank"):
        fit_pipls_core(X, Y, predictor_rank=3, n_components=2)


def test_predictor_svd_auto_rule_is_conservative() -> None:
    from pipls._core import _resolve_predictor_svd_solver

    assert (
        _resolve_predictor_svd_solver(
            shape=(500, 2000),
            predictor_rank=100,
            svd_solver="auto",
        )
        == "randomized"
    )
    assert (
        _resolve_predictor_svd_solver(
            shape=(500, 2000),
            predictor_rank=101,
            svd_solver="auto",
        )
        == "full"
    )
    assert (
        _resolve_predictor_svd_solver(
            shape=(499, 3000),
            predictor_rank=20,
            svd_solver="auto",
        )
        == "full"
    )
    assert (
        _resolve_predictor_svd_solver(
            shape=(500, 1999),
            predictor_rank=20,
            svd_solver="auto",
        )
        == "full"
    )


def test_explicit_predictor_svd_solver_overrides_auto_rule() -> None:
    from pipls._core import _resolve_predictor_svd_solver

    assert (
        _resolve_predictor_svd_solver(
            shape=(20, 10),
            predictor_rank=2,
            svd_solver="randomized",
        )
        == "randomized"
    )
    assert (
        _resolve_predictor_svd_solver(
            shape=(1000, 2000),
            predictor_rank=20,
            svd_solver="full",
        )
        == "full"
    )


def test_randomized_predictor_svd_is_reproducible_and_close_to_full() -> None:
    rng = np.random.default_rng(731)
    left, _ = np.linalg.qr(rng.normal(size=(90, 12)))
    right, _ = np.linalg.qr(rng.normal(size=(60, 12)))
    singular_values = np.array([30.0, 24.0, 19.0, 15.0, 12.0, 9.0, 5.0, 3.0, 2.0, 1.0, 0.5, 0.2])
    X = left @ np.diag(singular_values) @ right.T
    Y = X @ rng.normal(size=(60, 3)) + 0.01 * rng.normal(size=(90, 3))

    full = fit_pipls_core(
        X,
        Y,
        predictor_rank=6,
        n_components=2,
        svd_solver="full",
    )
    first = fit_pipls_core(
        X,
        Y,
        predictor_rank=6,
        n_components=2,
        svd_solver="randomized",
        random_state=17,
    )
    second = fit_pipls_core(
        X,
        Y,
        predictor_rank=6,
        n_components=2,
        svd_solver="randomized",
        random_state=17,
    )

    assert full.predictor_svd_solver == "full"
    assert full.x_rank_is_exact
    assert first.predictor_svd_solver == "randomized"
    assert not first.x_rank_is_exact
    np.testing.assert_allclose(first.Pi, second.Pi)
    np.testing.assert_allclose(
        first.standardized_regression_map,
        second.standardized_regression_map,
    )
    np.testing.assert_allclose(
        first.standardized_regression_map,
        full.standardized_regression_map,
        rtol=1e-6,
        atol=1e-8,
    )


def test_least_squares_randomized_predictor_svd_is_reproducible_and_close_to_full() -> None:
    rng = np.random.default_rng(20260824)
    left, _ = np.linalg.qr(rng.normal(size=(96, 14)))
    right, _ = np.linalg.qr(rng.normal(size=(72, 14)))
    singular_values = np.array(
        [40.0, 31.0, 25.0, 20.0, 16.0, 12.0, 9.0, 6.0, 4.0, 2.5, 1.5, 0.8, 0.4, 0.2]
    )
    X = left @ np.diag(singular_values) @ right.T
    Y = X @ rng.normal(size=(72, 4)) + 0.01 * rng.normal(size=(96, 4))

    full = fit_pipls_core(
        X,
        Y,
        predictor_rank=7,
        n_components=3,
        response_subspace="least_squares",
        svd_solver="full",
    )
    first = fit_pipls_core(
        X,
        Y,
        predictor_rank=7,
        n_components=3,
        response_subspace="least_squares",
        svd_solver="randomized",
        random_state=23,
    )
    second = fit_pipls_core(
        X,
        Y,
        predictor_rank=7,
        n_components=3,
        response_subspace="least_squares",
        svd_solver="randomized",
        random_state=23,
    )

    assert full.predictor_svd_solver == "full"
    assert first.predictor_svd_solver == "randomized"
    z_singular_values = np.linalg.svd(X @ first.Pi, compute_uv=False)
    assert np.count_nonzero(z_singular_values > first.rank_tolerance) == 7
    np.testing.assert_allclose(first.Pi, second.Pi)
    np.testing.assert_allclose(first.C @ first.C.T, second.C @ second.C.T)
    np.testing.assert_allclose(
        first.standardized_regression_map,
        second.standardized_regression_map,
    )
    np.testing.assert_allclose(
        first.standardized_regression_map,
        full.standardized_regression_map,
        rtol=1e-6,
        atol=1e-8,
    )


def test_randomized_predictor_svd_accepts_none_and_random_state() -> None:
    rng = np.random.default_rng(91)
    X = _center(rng.normal(size=(20, 8)))
    Y = _center(rng.normal(size=(20, 3)))

    none_result = fit_pipls_core(
        X,
        Y,
        predictor_rank=3,
        n_components=2,
        svd_solver="randomized",
        random_state=None,
    )
    first = fit_pipls_core(
        X,
        Y,
        predictor_rank=3,
        n_components=2,
        svd_solver="randomized",
        random_state=np.random.RandomState(5),
    )
    second = fit_pipls_core(
        X,
        Y,
        predictor_rank=3,
        n_components=2,
        svd_solver="randomized",
        random_state=np.random.RandomState(5),
    )

    assert none_result.predictor_svd_solver == "randomized"
    np.testing.assert_allclose(
        first.standardized_regression_map,
        second.standardized_regression_map,
    )
