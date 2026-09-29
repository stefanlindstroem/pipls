from typing import Any

import numpy as np
import pytest

from pipls import PiPLSRegression
from pipls.metrics import (
    neg_response_standardized_mse,
    response_standardized_mse,
)


def test_response_standardized_metric_uses_training_response_scale_when_scale_is_false() -> None:
    X = np.arange(24, dtype=np.float64).reshape(8, 3)
    y = np.array([0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0])
    model = PiPLSRegression(
        n_components=1,
        predictor_rank=1,
        scale=False,
    ).fit(X, y)

    prediction = model.predict(X)
    expected = np.mean(((y - prediction) / np.std(y, ddof=1)) ** 2)

    assert response_standardized_mse(model, X, y) == expected
    assert neg_response_standardized_mse(model, X, y) == -expected


def test_metric_callable_has_sklearn_scorer_signature() -> None:
    def consume_scorer(scorer: Any) -> None:
        assert callable(scorer)

    consume_scorer(neg_response_standardized_mse)


def test_training_response_scale_uses_sample_standard_deviation() -> None:
    from pipls.metrics import _training_response_scale

    y_train = np.array(
        [
            [1.0, 5.0],
            [3.0, 5.0],
            [5.0, 5.0],
        ]
    )

    scale = _training_response_scale(y_train)

    np.testing.assert_allclose(scale, np.array([2.0, 1.0]))


def test_training_response_scale_is_unit_for_singleton_training_data() -> None:
    from pipls.metrics import _training_response_scale

    np.testing.assert_array_equal(
        _training_response_scale(np.array([[2.0, -3.0]])),
        np.ones(2),
    )


def test_response_standardized_mse_is_uniform_over_samples_and_responses() -> None:
    from pipls.metrics import _response_standardized_mse

    y_true = np.array([[3.0, 9.0], [5.0, 13.0]])
    y_pred = np.array([[1.0, 5.0], [4.0, 9.0]])
    scale = np.array([2.0, 4.0])

    loss = _response_standardized_mse(y_true, y_pred, scale)

    expected = np.mean(np.array([1.0, 1.0, 0.25, 1.0]))
    assert loss == expected


def test_response_standardized_mse_supports_one_dimensional_targets() -> None:
    from pipls.metrics import _response_standardized_mse

    loss = _response_standardized_mse(
        np.array([2.0, 6.0]),
        np.array([0.0, 4.0]),
        np.array([2.0]),
    )

    assert loss == 1.0


def test_response_standardized_mse_rejects_invalid_scale() -> None:
    from pipls.metrics import _response_standardized_mse

    with pytest.raises(ValueError, match="positive finite"):
        _response_standardized_mse(
            np.ones((2, 2)),
            np.zeros((2, 2)),
            np.array([1.0, 0.0]),
        )
