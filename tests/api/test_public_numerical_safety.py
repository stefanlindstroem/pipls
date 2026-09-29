from __future__ import annotations

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from pipls import PiPLSRegression, PiPLSSearchCV
from pipls.metrics import response_standardized_mse


def _data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260722)
    X = rng.normal(size=(30, 5))
    Y = X @ rng.normal(size=(5, 2)) + 0.01 * rng.normal(size=(30, 2))
    return X, Y


def test_subnormal_scaled_data_fit_when_original_unit_map_is_finite() -> None:
    signal = np.linspace(-1.0, 1.0, 24)
    X = (signal * 1e-308).reshape(-1, 1)
    Y = (2.0 * signal * 1e-308).reshape(-1, 1)

    model = PiPLSRegression(n_components=1, predictor_rank=1).fit(X, Y)

    assert np.all(np.isfinite(model.coef_))
    assert model.coef_[0, 0] == pytest.approx(2.0, rel=1e-12, abs=0.0)
    assert np.all(np.isfinite(model.predict(X)))


def test_unrepresentable_original_unit_coefficients_raise_and_leave_no_fit() -> None:
    signal = np.linspace(-1.0, 1.0, 24)
    X = (signal * 1e-309).reshape(-1, 1)
    Y = signal.reshape(-1, 1)
    model = PiPLSRegression(n_components=1, predictor_rank=1)

    with pytest.raises(FloatingPointError, match="Fitted coef_"):
        model.fit(X, Y)

    with pytest.raises(NotFittedError):
        check_is_fitted(model)


def test_failed_fixed_refit_clears_previous_fitted_state() -> None:
    X, Y = _data()
    model = PiPLSRegression(n_components=1, predictor_rank=2).fit(X, Y)
    model.set_params(predictor_rank=99)

    with pytest.raises(ValueError, match="predictor_rank <= min"):
        model.fit(X, Y)

    with pytest.raises(NotFittedError):
        check_is_fitted(model)
    with pytest.raises(NotFittedError):
        model.predict(X)


def test_failed_path_refit_clears_previous_search_state() -> None:
    X, Y = _data()
    search = PiPLSSearchCV(
        n_components_values=[1],
        predictor_rank_values=[1],
        cv=3,
    ).fit(X, Y)
    search.set_params(max_predictor_rank=99, predictor_rank_values=[99])

    with pytest.raises(ValueError, match="predictor_rank_values"):
        search.fit(X, Y)

    with pytest.raises(NotFittedError):
        check_is_fitted(search)


def test_copy_false_accepts_read_only_arrays() -> None:
    X, Y = _data()
    X.setflags(write=False)
    Y.setflags(write=False)

    model = PiPLSRegression(
        n_components=1,
        predictor_rank=2,
        copy=False,
    ).fit(X, Y)

    assert np.all(np.isfinite(model.coef_))


def test_copy_false_protects_overlapping_predictor_and_response_memory() -> None:
    rng = np.random.default_rng(41)
    shared = rng.normal(size=(30, 6))
    X = shared[:, :4]
    Y = shared[:, 2:4]
    expected = PiPLSRegression(
        n_components=1,
        predictor_rank=2,
        copy=True,
    ).fit(X.copy(), Y.copy())

    actual = PiPLSRegression(
        n_components=1,
        predictor_rank=2,
        copy=False,
    ).fit(X, Y)

    np.testing.assert_allclose(actual.coef_, expected.coef_)
    np.testing.assert_allclose(actual.intercept_, expected.intercept_)


def test_unscaled_extreme_data_are_rejected_before_core_overflow() -> None:
    signal = np.linspace(-1.0, 1.0, 24)
    X = (signal * 1e300).reshape(-1, 1)
    Y = (signal * 1e300).reshape(-1, 1)
    model = PiPLSRegression(
        n_components=1,
        predictor_rank=1,
        scale=False,
    )

    with pytest.raises(FloatingPointError, match="cross-products may overflow"):
        model.fit(X, Y)
    with pytest.raises(NotFittedError):
        check_is_fitted(model)


def test_public_prediction_rejects_nonfinite_output() -> None:
    X = np.linspace(-1.0, 1.0, 20).reshape(-1, 1)
    Y = (2.0 * X).copy()
    model = PiPLSRegression(n_components=1, predictor_rank=1).fit(X, Y)

    with pytest.raises(FloatingPointError, match="Prediction"):
        model.predict(np.array([[np.finfo(np.float64).max]]))


def test_public_scorer_rejects_unrepresentable_mse() -> None:
    X = np.linspace(-1.0, 1.0, 20).reshape(-1, 1)
    Y = X.copy()
    model = PiPLSRegression(n_components=1, predictor_rank=1).fit(X, Y)

    with pytest.raises(ValueError, match="not representable"):
        response_standardized_mse(
            model,
            np.zeros((2, 1)),
            np.full((2, 1), np.finfo(np.float64).max),
        )


def test_low_level_linalg_failure_is_translated_and_fit_state_is_cleared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X, Y = _data()
    model = PiPLSRegression(n_components=1, predictor_rank=2)

    def fail(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise np.linalg.LinAlgError("synthetic failure")

    monkeypatch.setattr("pipls.regression.fit_pipls_core", fail)
    with pytest.raises(ValueError, match="numerical decomposition"):
        model.fit(X, Y)
    with pytest.raises(NotFittedError):
        check_is_fitted(model)
