from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from pipls._core import fit_pipls_core

_EXPECTED_DILATION = np.array([0.7318678674635507, 0.2860934054613716])
_EXPECTED_REGRESSION_MAP = np.array(
    [
        [0.0227515116210289, 0.0599525243839417, 0.0124157984852828],
        [-0.1621045910529089, -0.0923178863211663, -0.1302212944827721],
        [-0.5049202236976003, 0.0676477362676501, -0.4499077660476559],
        [-0.1229550816438217, 0.0125401683676110, -0.1090682976181716],
        [-0.0805912433251241, -0.2536619989695337, -0.0388296444243356],
    ]
)

# Frozen from an independent Choice-C calculation using the Gram solve on this
# deliberately well-conditioned fixture, not from fit_pipls_core().
_EXPECTED_LS_DILATION = np.array([0.7381881964004667, 0.2860137152705733])
_EXPECTED_LS_REGRESSION_MAP = np.array(
    [
        [0.0250104690774571, 0.0585010830360267, 0.0146760034100676],
        [-0.1399744640999856, -0.1040120610559612, -0.1189099886712646],
        [-0.4912209602342201, 0.0711780586674309, -0.4891006811999991],
        [-0.1026519881800129, 0.0021430415086542, -0.1001131938314765],
        [-0.0902488269283878, -0.2478252769440438, -0.0469117333277447],
    ]
)


def _independent_choice_c_reference(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    predictor_rank: int,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray]:
    _, _, x_vt = np.linalg.svd(X, full_matrices=False)
    Pi = x_vt[:predictor_rank, :].T
    Z = X @ Pi
    cross_product = Z.T @ Y
    gram = Z.T @ Z
    choice_c = cross_product.T @ np.linalg.solve(gram, cross_product)
    choice_c = 0.5 * (choice_c + choice_c.T)
    eigenvalues, eigenvectors = np.linalg.eigh(choice_c)
    order = np.argsort(eigenvalues)[::-1][:n_components]
    C = eigenvectors[:, order]
    W = np.linalg.lstsq(Z, Y @ C, rcond=None)[0]
    M, dilation, N_t = np.linalg.svd(W, full_matrices=False)
    P = Pi @ M[:, :n_components]
    Q = C @ N_t.T[:, :n_components]
    regression_map = P @ np.diag(dilation[:n_components]) @ Q.T
    return dilation[:n_components], regression_map


def test_core_matches_frozen_trusted_reference() -> None:
    rng = np.random.default_rng(20260715)
    X = rng.normal(size=(8, 5))
    Y = rng.normal(size=(8, 3))
    X -= X.mean(axis=0, keepdims=True)
    Y -= Y.mean(axis=0, keepdims=True)

    result = fit_pipls_core(X, Y, predictor_rank=3, n_components=2)

    assert_allclose(np.diag(result.D), _EXPECTED_DILATION, rtol=2e-13, atol=2e-13)
    assert_allclose(
        result.standardized_regression_map,
        _EXPECTED_REGRESSION_MAP,
        rtol=2e-13,
        atol=2e-13,
    )


def test_least_squares_core_matches_independent_frozen_reference() -> None:
    rng = np.random.default_rng(20260715)
    X = rng.normal(size=(8, 5))
    Y = rng.normal(size=(8, 3))
    X -= X.mean(axis=0, keepdims=True)
    Y -= Y.mean(axis=0, keepdims=True)

    independent_dilation, independent_map = _independent_choice_c_reference(
        X,
        Y,
        predictor_rank=3,
        n_components=2,
    )
    result = fit_pipls_core(
        X,
        Y,
        predictor_rank=3,
        n_components=2,
        response_subspace="least_squares",
    )

    assert_allclose(
        independent_dilation,
        _EXPECTED_LS_DILATION,
        rtol=2e-13,
        atol=2e-13,
    )
    assert_allclose(
        independent_map,
        _EXPECTED_LS_REGRESSION_MAP,
        rtol=2e-13,
        atol=2e-13,
    )
    assert_allclose(
        np.diag(result.D),
        _EXPECTED_LS_DILATION,
        rtol=2e-13,
        atol=2e-13,
    )
    assert_allclose(
        result.standardized_regression_map,
        _EXPECTED_LS_REGRESSION_MAP,
        rtol=2e-13,
        atol=2e-13,
    )
