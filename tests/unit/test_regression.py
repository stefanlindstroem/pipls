import pickle

import numpy as np
import pytest
from sklearn.base import clone

from pipls import PiPLSRegression


def _data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(12)
    X = rng.normal(size=(40, 8))
    B = rng.normal(size=(8, 3))
    Y = X @ B + 0.05 * rng.normal(size=(40, 3))
    return X, Y


def test_fit_exposes_expected_fixed_rank_attributes() -> None:
    X, Y = _data()
    model = PiPLSRegression(n_components=2, predictor_rank=4).fit(X, Y)

    assert model.predictor_rank == 4
    assert model.max_predictor_rank_ == min(X.shape[1], X.shape[0] - 1)
    assert model.decomposition_.predictor_directions.shape == (8, 2)
    assert model.decomposition_.dilation.shape == (2,)
    assert model.decomposition_.response_directions.shape == (3, 2)
    assert model.coef_.shape == (3, 8)
    assert model.intercept_.shape == (3,)
    assert model.x_scores_.shape == (40, 2)
    assert model.y_scores_.shape == (40, 2)


def test_prediction_matches_affine_attributes() -> None:
    X, Y = _data()
    model = PiPLSRegression(n_components=2, predictor_rank=4).fit(X, Y)

    np.testing.assert_allclose(model.predict(X), X @ model.coef_.T + model.intercept_)


def test_transform_uses_fitted_centering_and_scaling() -> None:
    X, Y = _data()
    model = PiPLSRegression(n_components=2, predictor_rank=4).fit(X, Y)

    x_scores, y_scores = model.transform(X, Y)
    np.testing.assert_allclose(x_scores, model.x_scores_)
    np.testing.assert_allclose(y_scores, model.y_scores_)


def test_scale_false_still_centers() -> None:
    X, Y = _data()
    model = PiPLSRegression(
        n_components=2,
        predictor_rank=4,
        scale=False,
    ).fit(X, Y)

    np.testing.assert_allclose(model.x_scale_, 1.0)
    np.testing.assert_allclose(model.y_scale_, 1.0)
    np.testing.assert_allclose(np.mean(model.x_scores_, axis=0), 0.0, atol=1e-12)


@pytest.mark.parametrize(
    ("scale_x", "scale_y", "expect_x_scaling", "expect_y_scaling"),
    [
        (False, False, False, False),
        (False, True, False, True),
        (True, False, True, False),
        (True, True, True, True),
    ],
)
def test_predictor_and_response_scaling_can_be_controlled_independently(
    scale_x: bool,
    scale_y: bool,
    expect_x_scaling: bool,
    expect_y_scaling: bool,
) -> None:
    X, Y = _data()
    model = PiPLSRegression(
        n_components=2,
        predictor_rank=4,
        scale=False,
        scale_x=scale_x,
        scale_y=scale_y,
    ).fit(X, Y)

    expected_x_scale = np.std(X, axis=0, ddof=1) if expect_x_scaling else np.ones(X.shape[1])
    expected_y_scale = np.std(Y, axis=0, ddof=1) if expect_y_scaling else np.ones(Y.shape[1])
    np.testing.assert_allclose(model.x_scale_, expected_x_scale)
    np.testing.assert_allclose(model.y_scale_, expected_y_scale)


@pytest.mark.parametrize("legacy_scale", [False, True])
def test_independent_scaling_overrides_preserve_legacy_scale_behavior(
    legacy_scale: bool,
) -> None:
    X, Y = _data()
    legacy = PiPLSRegression(
        n_components=2,
        predictor_rank=4,
        scale=legacy_scale,
        svd_solver="full",
    ).fit(X, Y)
    overridden = PiPLSRegression(
        n_components=2,
        predictor_rank=4,
        scale=not legacy_scale,
        scale_x=legacy_scale,
        scale_y=legacy_scale,
        svd_solver="full",
    ).fit(X, Y)

    np.testing.assert_allclose(overridden.x_scale_, legacy.x_scale_)
    np.testing.assert_allclose(overridden.y_scale_, legacy.y_scale_)
    np.testing.assert_allclose(overridden.coef_, legacy.coef_)
    np.testing.assert_allclose(overridden.intercept_, legacy.intercept_)
    np.testing.assert_allclose(overridden.predict(X), legacy.predict(X))


def test_one_dimensional_response_round_trips_as_one_dimensional() -> None:
    X, Y = _data()
    model = PiPLSRegression(n_components=1, predictor_rank=3).fit(X, Y[:, 0])

    assert model.predict(X).shape == (X.shape[0],)
    assert isinstance(model.score(X, Y[:, 0]), float)


def test_invalid_constructor_combination_is_rejected() -> None:
    X, Y = _data()
    with pytest.raises(ValueError, match="n_components <= predictor_rank"):
        PiPLSRegression(n_components=3, predictor_rank=2).fit(X, Y)


def test_wrong_feature_count_is_rejected_at_prediction() -> None:
    X, Y = _data()
    model = PiPLSRegression(n_components=2, predictor_rank=4).fit(X, Y)

    with pytest.raises(ValueError, match="expecting 8 features"):
        model.predict(X[:, :-1])


def test_constant_columns_have_unit_scale() -> None:
    X, Y = _data()
    X[:, 0] = 3.0
    Y[:, 0] = -2.0
    model = PiPLSRegression(n_components=1, predictor_rank=3).fit(X, Y)

    assert model.x_scale_[0] == 1.0
    assert model.y_scale_[0] == 1.0


def test_fixed_rank_pair_is_required_and_keyword_only() -> None:
    with pytest.raises(TypeError):
        PiPLSRegression()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        PiPLSRegression(n_components=1)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        PiPLSRegression(predictor_rank=1)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        PiPLSRegression(1, predictor_rank=1)  # type: ignore[call-arg]


def test_response_subspace_default_and_clone_contract() -> None:
    model = PiPLSRegression(n_components=2, predictor_rank=4)

    assert model.response_subspace == "cross_covariance"
    assert model.get_params()["response_subspace"] == "cross_covariance"
    cloned = clone(
        PiPLSRegression(
            n_components=2,
            predictor_rank=4,
            response_subspace="least_squares",
        )
    )
    assert cloned.response_subspace == "least_squares"
    assert not hasattr(cloned, "coef_")


def test_explicit_cross_covariance_matches_default_fit_exactly() -> None:
    X, Y = _data()
    default = PiPLSRegression(
        n_components=2,
        predictor_rank=4,
        svd_solver="full",
    ).fit(X, Y)
    explicit = PiPLSRegression(
        n_components=2,
        predictor_rank=4,
        response_subspace="cross_covariance",
        svd_solver="full",
    ).fit(X, Y)

    np.testing.assert_array_equal(
        explicit.decomposition_.predictor_directions,
        default.decomposition_.predictor_directions,
    )
    np.testing.assert_array_equal(
        explicit.decomposition_.dilation,
        default.decomposition_.dilation,
    )
    np.testing.assert_array_equal(
        explicit.decomposition_.response_directions,
        default.decomposition_.response_directions,
    )
    np.testing.assert_array_equal(explicit.coef_, default.coef_)
    np.testing.assert_array_equal(explicit.intercept_, default.intercept_)
    np.testing.assert_array_equal(explicit.predict(X), default.predict(X))


def test_least_squares_response_subspace_supports_public_fit_surfaces() -> None:
    X, Y = _data()
    model = PiPLSRegression(
        n_components=2,
        predictor_rank=4,
        response_subspace="least_squares",
        svd_solver="full",
    ).fit(X, Y)

    prediction = model.predict(X)
    x_scores, y_scores = model.transform(X, Y)
    assert model.response_subspace == "least_squares"
    assert model.decomposition_.predictor_directions.shape == (X.shape[1], 2)
    assert model.decomposition_.response_directions.shape == (Y.shape[1], 2)
    assert prediction.shape == Y.shape
    assert x_scores.shape == (X.shape[0], 2)
    assert y_scores.shape == (X.shape[0], 2)
    assert np.all(np.isfinite(prediction))
    assert np.all(np.isfinite(model.coef_))


@pytest.mark.parametrize("value", ["xcov", "lstsq", "var", 1, None])
def test_invalid_response_subspace_is_rejected_without_fitted_state(value: object) -> None:
    X, Y = _data()
    model = PiPLSRegression(
        n_components=2,
        predictor_rank=4,
        response_subspace=value,  # type: ignore[arg-type]
    )

    with pytest.raises(
        ValueError,
        match='response_subspace must be "cross_covariance" or "least_squares"',
    ):
        model.fit(X, Y)

    assert not hasattr(model, "coef_")
    assert not hasattr(model, "decomposition_")


@pytest.mark.parametrize(
    ("scale_x", "scale_y"),
    [
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ],
)
def test_least_squares_response_subspace_supports_all_scaling_combinations(
    scale_x: bool,
    scale_y: bool,
) -> None:
    X, Y = _data()
    model = PiPLSRegression(
        n_components=2,
        predictor_rank=4,
        response_subspace="least_squares",
        scale=False,
        scale_x=scale_x,
        scale_y=scale_y,
        svd_solver="full",
    ).fit(X, Y)

    expected_x_scale = (
        np.std(X, axis=0, ddof=1) if scale_x else np.ones(X.shape[1])
    )
    expected_y_scale = (
        np.std(Y, axis=0, ddof=1) if scale_y else np.ones(Y.shape[1])
    )
    np.testing.assert_allclose(model.x_scale_, expected_x_scale)
    np.testing.assert_allclose(model.y_scale_, expected_y_scale)
    expected_coef_matrix = (
        model.decomposition_.standardized_regression_map
        * model.y_scale_[None, :]
        / model.x_scale_[:, None]
    )
    np.testing.assert_allclose(model.coef_.T, expected_coef_matrix)
    assert np.all(np.isfinite(model.predict(X)))
    assert np.all(np.isfinite(model.x_scores_))
    assert np.all(np.isfinite(model.y_scores_))


def test_fitted_least_squares_estimator_pickle_round_trip() -> None:
    X, Y = _data()
    model = PiPLSRegression(
        n_components=2,
        predictor_rank=4,
        response_subspace="least_squares",
        scale_x=True,
        scale_y=False,
        svd_solver="full",
    ).fit(X, Y)

    restored = pickle.loads(pickle.dumps(model))

    assert restored.response_subspace == "least_squares"
    np.testing.assert_array_equal(restored.coef_, model.coef_)
    np.testing.assert_array_equal(restored.intercept_, model.intercept_)
    np.testing.assert_array_equal(restored.predict(X), model.predict(X))
    np.testing.assert_array_equal(
        restored.decomposition_.standardized_regression_map,
        model.decomposition_.standardized_regression_map,
    )


def test_set_params_response_subspace_refit_matches_fresh_estimator() -> None:
    X, Y = _data()
    model = PiPLSRegression(
        n_components=2,
        predictor_rank=4,
        response_subspace="cross_covariance",
        svd_solver="full",
    ).fit(X, Y)

    model.set_params(response_subspace="least_squares").fit(X, Y)
    fresh = PiPLSRegression(
        n_components=2,
        predictor_rank=4,
        response_subspace="least_squares",
        svd_solver="full",
    ).fit(X, Y)

    assert model.response_subspace == "least_squares"
    np.testing.assert_array_equal(model.coef_, fresh.coef_)
    np.testing.assert_array_equal(model.intercept_, fresh.intercept_)
    np.testing.assert_array_equal(model.predict(X), fresh.predict(X))
    np.testing.assert_array_equal(
        model.decomposition_.standardized_regression_map,
        fresh.decomposition_.standardized_regression_map,
    )

def test_fixed_fit_creates_no_search_state() -> None:
    X, Y = _data()
    model = PiPLSRegression(n_components=2, predictor_rank=4).fit(X, Y)

    assert not hasattr(model, "selection_")
    assert not hasattr(model, "cv_results_")
    assert not hasattr(model, "oof_report_")


def test_small_auto_svd_uses_full_solver_and_reports_exact_rank() -> None:
    X, Y = _data()
    model = PiPLSRegression(
        n_components=2,
        predictor_rank=3,
        svd_solver="auto",
    ).fit(X, Y)

    assert model.decomposition_.predictor_svd_solver == "full"
    assert model.decomposition_.predictor_numerical_rank_is_exact


def test_randomized_svd_estimator_is_reproducible_and_close_to_full() -> None:
    rng = np.random.default_rng(817)
    left, _ = np.linalg.qr(rng.normal(size=(100, 14)))
    right, _ = np.linalg.qr(rng.normal(size=(70, 14)))
    values = np.array(
        [35.0, 28.0, 21.0, 16.0, 12.0, 9.0, 6.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.2, 0.1]
    )
    X = left @ np.diag(values) @ right.T
    Y = X @ rng.normal(size=(70, 3)) + 0.01 * rng.normal(size=(100, 3))

    full = PiPLSRegression(
        n_components=2,
        predictor_rank=6,
        svd_solver="full",
        random_state=None,
    ).fit(X, Y)
    first = PiPLSRegression(
        n_components=2,
        predictor_rank=6,
        svd_solver="randomized",
        random_state=23,
    ).fit(X, Y)
    second = PiPLSRegression(
        n_components=2,
        predictor_rank=6,
        svd_solver="randomized",
        random_state=23,
    ).fit(X, Y)

    assert first.decomposition_.predictor_svd_solver == "randomized"
    assert not first.decomposition_.predictor_numerical_rank_is_exact
    np.testing.assert_allclose(first.coef_, second.coef_)
    np.testing.assert_allclose(first.predict(X), second.predict(X))
    np.testing.assert_allclose(first.predict(X), full.predict(X), rtol=1e-6, atol=1e-8)


def test_randomized_svd_accepts_none_and_random_state_instances() -> None:
    X, Y = _data()
    none_model = PiPLSRegression(
        n_components=2,
        predictor_rank=3,
        svd_solver="randomized",
        random_state=None,
    ).fit(X, Y)
    first = PiPLSRegression(
        n_components=2,
        predictor_rank=3,
        svd_solver="randomized",
        random_state=np.random.RandomState(23),
    ).fit(X, Y)
    second = PiPLSRegression(
        n_components=2,
        predictor_rank=3,
        svd_solver="randomized",
        random_state=np.random.RandomState(23),
    ).fit(X, Y)

    assert none_model.decomposition_.predictor_svd_solver == "randomized"
    np.testing.assert_allclose(first.coef_, second.coef_)
