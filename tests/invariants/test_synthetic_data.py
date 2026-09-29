from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from pipls.datasets import make_synthetic_data


def test_noise_free_signal_has_declared_predictor_and_response_ranks() -> None:
    X, Y = make_synthetic_data(
        n_samples=80,
        n_features=9,
        n_targets=6,
        n_shared=2,
        n_predictor_specific=3,
        n_response_specific=2,
        noise=0.0,
        random_state=123,
    )

    assert np.linalg.matrix_rank(X) == 5
    assert np.linalg.matrix_rank(Y) == 4


def test_synthetic_data_supports_structurally_absent_blocks() -> None:
    X, Y = make_synthetic_data(
        n_samples=6,
        n_features=4,
        n_targets=3,
        n_shared=0,
        n_predictor_specific=0,
        n_response_specific=0,
        noise=0.0,
        random_state=11,
    )

    assert_allclose(X, 0.0)
    assert_allclose(Y, 0.0)


def test_synthetic_generator_does_not_mutate_numpy_global_rng() -> None:
    np.random.seed(318)
    expected = np.random.random(5)
    np.random.seed(318)

    make_synthetic_data(
        n_samples=10,
        n_features=4,
        n_targets=3,
        n_shared=1,
        random_state=8,
    )
    observed = np.random.random(5)

    assert_allclose(observed, expected)
