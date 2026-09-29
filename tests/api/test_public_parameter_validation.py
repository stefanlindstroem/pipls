from __future__ import annotations

import warnings

import numpy as np
import pytest

from pipls import PiPLSRegression, PredictorRankSupportWarning

_MAX_RANDOM_STATE = int(np.iinfo(np.uint32).max)


def _data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260716)
    X = rng.normal(size=(24, 6))
    Y = X @ rng.normal(size=(6, 3)) + 0.02 * rng.normal(size=(24, 3))
    return X, Y


@pytest.mark.parametrize(
    "value",
    [0, -1, True, np.bool_(False), 1.0, np.float64(1.0), "1", None, np.nan, np.inf, []],
)
def test_n_components_rejects_nonpositive_and_noninteger_values(value: object) -> None:
    X, Y = _data()
    with pytest.raises(ValueError, match="n_components must be a positive integer"):
        PiPLSRegression(n_components=value, predictor_rank=2).fit(X, Y)  # type: ignore[arg-type]


def test_n_components_accepts_numpy_integer() -> None:
    X, Y = _data()
    model = PiPLSRegression(n_components=np.int64(2), predictor_rank=3).fit(X, Y)
    assert model.x_rotations_.shape[1] == 2


def test_n_components_rejects_more_components_than_response_columns() -> None:
    X, Y = _data()
    with pytest.raises(ValueError, match="number of response columns"):
        PiPLSRegression(n_components=4, predictor_rank=4).fit(X, Y)


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
        True,
        np.bool_(True),
        2.0,
        np.float64(2.0),
        "2",
        "invalid",
        None,
    ],
)
def test_predictor_rank_rejects_nonpositive_noninteger_and_mode_values(value: object) -> None:
    X, Y = _data()
    with pytest.raises(ValueError, match="predictor_rank must be a positive integer"):
        PiPLSRegression(n_components=1, predictor_rank=value).fit(X, Y)  # type: ignore[arg-type]


def test_predictor_rank_accepts_numpy_integer() -> None:
    X, Y = _data()
    model = PiPLSRegression(n_components=2, predictor_rank=np.int64(3)).fit(X, Y)
    assert model.predictor_rank == 3


def test_predictor_rank_rejects_rank_above_matrix_dimensions() -> None:
    X, Y = _data()
    with pytest.raises(ValueError, match="predictor_rank <= min"):
        PiPLSRegression(n_components=1, predictor_rank=7).fit(X, Y)


def test_predictor_rank_rejects_rank_above_numerical_rank() -> None:
    X, Y = _data()
    X[:, 2:] = X[:, :1]
    with pytest.raises(ValueError, match="numerical rank"):
        PiPLSRegression(n_components=1, predictor_rank=3).fit(X, Y)


def test_support_warning_boundary_is_strictly_below_three() -> None:
    X, Y = _data()
    rng = np.random.default_rng(13)
    X_wide = np.column_stack([X, rng.normal(size=(X.shape[0], 3))])

    with warnings.catch_warnings():
        warnings.simplefilter("error", PredictorRankSupportWarning)
        PiPLSRegression(n_components=1, predictor_rank=8).fit(X_wide, Y)

    with pytest.warns(PredictorRankSupportWarning):
        PiPLSRegression(n_components=1, predictor_rank=9).fit(X_wide, Y)


@pytest.mark.parametrize(
    "value",
    [-1, True, np.bool_(False), 1.0, np.float64(1.0), "1", _MAX_RANDOM_STATE + 1, []],
)
def test_random_state_rejects_invalid_or_out_of_range_values(value: object) -> None:
    X, Y = _data()
    with pytest.raises(ValueError, match="random_state"):
        PiPLSRegression(
            n_components=1,
            predictor_rank=2,
            svd_solver="auto",
            random_state=value,  # type: ignore[arg-type]
        ).fit(X, Y)


@pytest.mark.parametrize("solver", ["auto", "randomized", "full"])
def test_random_state_none_is_accepted_for_all_solver_policies(solver: str) -> None:
    X, Y = _data()
    model = PiPLSRegression(
        n_components=1,
        predictor_rank=2,
        svd_solver=solver,  # type: ignore[arg-type]
        random_state=None,
    ).fit(X, Y)
    assert model.decomposition_.predictor_svd_solver in ("full", "randomized")


def test_random_state_instance_is_accepted() -> None:
    X, Y = _data()
    model = PiPLSRegression(
        n_components=1,
        predictor_rank=2,
        svd_solver="randomized",
        random_state=np.random.RandomState(7),
    ).fit(X, Y)
    assert model.decomposition_.predictor_svd_solver == "randomized"


@pytest.mark.parametrize("parameter", ["scale", "copy"])
@pytest.mark.parametrize("value", [0, 1, 1.0, "true", None, []])
def test_boolean_parameters_reject_nonboolean_values(parameter: str, value: object) -> None:
    X, Y = _data()
    kwargs = {parameter: value}
    with pytest.raises(ValueError, match=f"{parameter} must be boolean"):
        PiPLSRegression(
            n_components=1,
            predictor_rank=2,
            **kwargs,  # type: ignore[arg-type]
        ).fit(X, Y)


@pytest.mark.parametrize("parameter", ["scale_x", "scale_y"])
@pytest.mark.parametrize("value", [0, 1, 1.0, "true", []])
def test_optional_boolean_parameters_reject_nonboolean_values(
    parameter: str,
    value: object,
) -> None:
    X, Y = _data()
    kwargs = {parameter: value}
    with pytest.raises(ValueError, match=f"{parameter} must be boolean or None"):
        PiPLSRegression(
            n_components=1,
            predictor_rank=2,
            **kwargs,  # type: ignore[arg-type]
        ).fit(X, Y)


@pytest.mark.parametrize("parameter", ["scale_x", "scale_y"])
def test_optional_boolean_parameters_accept_none(parameter: str) -> None:
    X, Y = _data()
    model = PiPLSRegression(
        n_components=1,
        predictor_rank=2,
        **{parameter: None},  # type: ignore[arg-type]
    ).fit(X, Y)
    assert model.predictor_rank == 2


def test_numpy_boolean_parameters_are_accepted() -> None:
    X, Y = _data()
    model = PiPLSRegression(
        n_components=1,
        predictor_rank=2,
        scale=np.bool_(True),
        scale_x=np.bool_(False),
        scale_y=np.bool_(True),
        copy=np.bool_(False),
    ).fit(X, Y)
    assert model.predictor_rank == 2


@pytest.mark.parametrize("value", [1, None, [], np.asarray(["full"])])
def test_svd_solver_rejects_nonstring_values_cleanly(value: object) -> None:
    X, Y = _data()
    with pytest.raises(ValueError, match="svd_solver"):
        PiPLSRegression(
            n_components=1,
            predictor_rank=2,
            svd_solver=value,  # type: ignore[arg-type]
        ).fit(X, Y)
