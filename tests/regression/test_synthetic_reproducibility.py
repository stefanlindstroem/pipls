from __future__ import annotations

import numpy as np
from numpy.testing import assert_array_equal

from pipls.datasets import make_synthetic_data

PARAMETERS = {
    "n_samples": 17,
    "n_features": 6,
    "n_targets": 5,
    "n_shared": 2,
    "n_predictor_specific": 1,
    "n_response_specific": 2,
    "noise": (0.15, 0.25),
}


def test_different_seed_changes_generated_values() -> None:
    X_first, Y_first = make_synthetic_data(**PARAMETERS, random_state=314)
    X_second, Y_second = make_synthetic_data(**PARAMETERS, random_state=315)

    assert not np.array_equal(X_first, X_second)
    assert not np.array_equal(Y_first, Y_second)


def test_synthetic_generator_is_exactly_reproducible() -> None:
    X_first, Y_first = make_synthetic_data(**PARAMETERS, random_state=271)
    X_second, Y_second = make_synthetic_data(
        **PARAMETERS,
        random_state=np.int64(271),
    )

    assert_array_equal(X_first, X_second)
    assert_array_equal(Y_first, Y_second)
