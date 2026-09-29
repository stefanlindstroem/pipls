from __future__ import annotations

import numpy as np

from pipls import PiPLSRegression
from pipls.inspection import pipls_display_factors


def test_display_sign_canonicalization_preserves_fitted_regression_map() -> None:
    rng = np.random.default_rng(1204)
    X = rng.normal(size=(24, 6))
    Y = rng.normal(size=(24, 3))
    model = PiPLSRegression(n_components=2, predictor_rank=3).fit(X, Y)
    decomposition = model.decomposition_
    predictor_before = decomposition.predictor_directions.copy()
    response_before = decomposition.response_directions.copy()

    factor_views = (
        pipls_display_factors(decomposition),
        pipls_display_factors(
            decomposition,
            response_index=1,
            response_sign="negative",
        ),
    )
    for factors in factor_views:
        display_map = (
            factors.predictor_directions
            @ np.diag(factors.dilation)
            @ factors.response_directions.T
        )

        np.testing.assert_allclose(
            display_map,
            decomposition.standardized_regression_map,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            factors.predictor_directions @ factors.weighted_response_directions.T,
            decomposition.standardized_regression_map,
            atol=1e-14,
        )
    np.testing.assert_array_equal(decomposition.predictor_directions, predictor_before)
    np.testing.assert_array_equal(decomposition.response_directions, response_before)
    assert model.x_rotations_ is decomposition.predictor_directions
    assert model.y_rotations_ is decomposition.response_directions


def test_balanced_biplot_preserves_selected_score_loading_reconstruction() -> None:
    from pipls.inspection import biplot_coordinates, latent_structure

    rng = np.random.default_rng(2301)
    X = rng.normal(size=(48, 9))
    Y = X[:, :4] @ rng.normal(size=(4, 3)) + 0.05 * rng.normal(size=(48, 3))
    model = PiPLSRegression(
        n_components=3,
        predictor_rank=6,
        svd_solver="full",
    ).fit(X, Y)
    structure = latent_structure(model)

    coordinates = biplot_coordinates(structure, components=(0, 2))

    expected = structure.x_scores[:, [0, 2]] @ structure.x_loadings[:, [0, 2]].T
    actual = coordinates.sample_coordinates @ coordinates.predictor_coordinates.T
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(
        np.linalg.norm(coordinates.sample_coordinates, axis=0),
        np.linalg.norm(coordinates.predictor_coordinates, axis=0),
        rtol=1e-13,
        atol=1e-13,
    )
