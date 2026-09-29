from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import check_estimator

import pipls
from pipls import PiPLSRegression, PiPLSSearchCV
from pipls.component_path import PiPLSComponentPath, PiPLSSelection
from pipls.decomposition import PiPLSDecomposition


def _data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260718)
    X = rng.normal(size=(48, 7))
    Y = X @ rng.normal(size=(7, 3)) + 0.05 * rng.normal(size=(48, 3))
    return X, Y


def _fixed_estimator() -> PiPLSRegression:
    return PiPLSRegression(
        n_components=2,
        predictor_rank=3,
        scale=False,
        svd_solver="full",
        random_state=None,
    )


def test_search_cv_is_public_selection_class() -> None:
    assert pipls.PiPLSSearchCV is PiPLSSearchCV


def test_fixed_regression_and_path_configuration_have_distinct_ownership() -> None:
    regression = _fixed_estimator()
    search = PiPLSSearchCV()

    assert regression.n_components == 2
    assert regression.predictor_rank == 3
    assert not hasattr(regression, "cv")
    assert not hasattr(regression, "samples_per_predictor_rank")
    assert search.samples_per_predictor_rank == 10.0
    assert search.cv == 5
    assert search.n_components_values == "all"
    assert search.scoring == "neg_response_standardized_mse"
    assert callable(search.select)
    assert callable(search.refit)
    assert callable(search.oof_report)


def test_search_cv_clones_with_default_configuration() -> None:
    search = PiPLSSearchCV()
    cloned = clone(search)

    assert cloned.get_params(deep=False) == search.get_params(deep=False)
    assert callable(cloned.select)
    assert callable(cloned.refit)
    assert callable(cloned.oof_report)


def test_selection_method_signatures_separate_component_and_rank_tolerances() -> None:
    for method in (PiPLSSearchCV.select, PiPLSSearchCV.refit):
        signature = inspect.signature(method)
        assert signature.parameters["relative_tolerance"].default is None
        assert np.isposinf(signature.parameters["absolute_tolerance"].default)
        assert signature.parameters["relative_tolerance"].kind is (
            inspect.Parameter.KEYWORD_ONLY
        )
        assert signature.parameters["absolute_tolerance"].kind is (
            inspect.Parameter.KEYWORD_ONLY
        )

    refit_signature = inspect.signature(PiPLSSearchCV.refit)
    selection_parameter = refit_signature.parameters["selection"]
    assert selection_parameter.default is None
    assert selection_parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert selection_parameter.annotation == "PiPLSSelection | None"
    assert "selection" not in inspect.signature(PiPLSSearchCV.select).parameters

    parameters = PiPLSSearchCV().get_params(deep=False)
    assert "relative_tolerance" not in parameters
    assert "absolute_tolerance" not in parameters
    assert parameters["predictor_rank_relative_tolerance"] is None
    assert np.isposinf(parameters["predictor_rank_absolute_tolerance"])

    constructor = inspect.signature(PiPLSSearchCV)
    assert constructor.parameters["predictor_rank_relative_tolerance"].default is None
    assert np.isposinf(
        constructor.parameters["predictor_rank_absolute_tolerance"].default
    )


def test_fixed_regression_constructor_requires_keyword_only_rank_pair() -> None:
    signature = inspect.signature(PiPLSRegression)
    for name in ("n_components", "predictor_rank"):
        parameter = signature.parameters[name]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is inspect.Parameter.empty
    assert signature.parameters["response_subspace"].default == "cross_covariance"


def test_search_owns_selection_without_estimator_delegation() -> None:
    search = PiPLSSearchCV()

    assert callable(search.select)
    assert callable(search.refit)
    assert callable(search.oof_report)
    for method_name in (
        "predict",
        "transform",
        "fit_transform",
        "inverse_transform",
        "score",
        "get_feature_names_out",
        "set_output",
    ):
        assert not hasattr(search, method_name)
    assert hasattr(_fixed_estimator(), "set_output")


def test_fixed_estimator_interoperates_with_grid_search_for_explicit_pairs() -> None:
    X, Y = _data()
    search = GridSearchCV(
        _fixed_estimator(),
        param_grid=[
            {"n_components": [1], "predictor_rank": [2]},
            {"n_components": [2], "predictor_rank": [3]},
        ],
        cv=3,
    ).fit(X, Y)

    assert isinstance(search.best_estimator_, PiPLSRegression)
    assert search.best_estimator_.predictor_rank in (2, 3)



def test_grid_search_preserves_least_squares_response_subspace_configuration() -> None:
    X, Y = _data()
    template = _fixed_estimator().set_params(response_subspace="least_squares")
    search = GridSearchCV(
        template,
        param_grid=[
            {"n_components": [1], "predictor_rank": [2]},
            {"n_components": [2], "predictor_rank": [3]},
        ],
        cv=3,
    ).fit(X, Y)

    assert template.response_subspace == "least_squares"
    assert not hasattr(template, "coef_")
    assert isinstance(search.best_estimator_, PiPLSRegression)
    assert search.best_estimator_.response_subspace == "least_squares"
    assert search.best_estimator_.predictor_rank in (2, 3)


def test_pls_style_method_signatures_include_copy_controls() -> None:
    assert tuple(inspect.signature(PiPLSRegression.predict).parameters) == (
        "self",
        "X",
        "copy",
    )
    assert tuple(inspect.signature(PiPLSRegression.transform).parameters) == (
        "self",
        "X",
        "y",
        "copy",
    )
    assert tuple(inspect.signature(PiPLSRegression.fit_transform).parameters) == (
        "self",
        "X",
        "y",
    )


def test_fit_transform_matches_pls_style_pair_of_scores() -> None:
    X, Y = _data()
    model = _fixed_estimator()

    x_scores, y_scores = model.fit_transform(X, Y)
    transformed_x, transformed_y = model.transform(X, Y)

    np.testing.assert_allclose(x_scores, transformed_x)
    np.testing.assert_allclose(y_scores, transformed_y)
    np.testing.assert_allclose(x_scores, model.x_scores_)
    np.testing.assert_allclose(y_scores, model.y_scores_)


def test_standard_pls_style_attributes_have_documented_meaning() -> None:
    X, Y = _data()
    model = _fixed_estimator().fit(X, Y)
    X_cs = (X - model.x_mean_) / model.x_scale_
    Y_cs = (Y - model.y_mean_) / model.y_scale_

    np.testing.assert_allclose(model.x_scores_, X_cs @ model.x_rotations_)
    np.testing.assert_allclose(model.y_scores_, Y_cs @ model.y_rotations_)
    assert model.x_loadings_.shape == (X.shape[1], model.n_components)
    assert model.y_loadings_.shape == (Y.shape[1], model.n_components)

    x_residual = X_cs - model.x_scores_ @ model.x_loadings_.T
    y_residual = Y_cs - model.y_scores_ @ model.y_loadings_.T
    np.testing.assert_allclose(model.x_scores_.T @ x_residual, 0.0, atol=1e-10)
    np.testing.assert_allclose(model.y_scores_.T @ y_residual, 0.0, atol=1e-10)


def test_public_decomposition_is_read_only_and_matches_fitted_rotations() -> None:
    X, Y = _data()
    model = _fixed_estimator().fit(X, Y)
    decomposition = model.decomposition_

    assert isinstance(decomposition, PiPLSDecomposition)
    for matrix in (
        decomposition.predictor_directions,
        decomposition.dilation,
        decomposition.response_directions,
    ):
        assert not matrix.flags.writeable
    standardized_map = decomposition.standardized_regression_map
    assert not standardized_map.flags.writeable
    np.testing.assert_allclose(
        standardized_map,
        (decomposition.predictor_directions * decomposition.dilation[None, :])
        @ decomposition.response_directions.T,
    )
    assert model.x_rotations_ is decomposition.predictor_directions
    assert model.y_rotations_ is decomposition.response_directions


    with pytest.raises(ValueError, match="read-only"):
        decomposition.predictor_directions[0, 0] = 0.0
    with pytest.raises(FrozenInstanceError):
        decomposition.predictor_numerical_rank = 0  # type: ignore[misc]


def test_copy_parameter_matches_pls_fit_semantics() -> None:
    X, Y = _data()
    X_original = X.copy()
    Y_original = Y.copy()
    _fixed_estimator().set_params(copy=True).fit(X, Y)
    np.testing.assert_array_equal(X, X_original)
    np.testing.assert_array_equal(Y, Y_original)

    X_mutable = X_original.copy()
    Y_mutable = Y_original.copy()
    _fixed_estimator().set_params(copy=False).fit(X_mutable, Y_mutable)
    assert not np.array_equal(X_mutable, X_original)
    assert not np.array_equal(Y_mutable, Y_original)
    np.testing.assert_allclose(np.mean(X_mutable, axis=0), 0.0, atol=1e-14)
    np.testing.assert_allclose(np.mean(Y_mutable, axis=0), 0.0, atol=1e-14)


def test_feature_names_and_set_output_match_sklearn_transformers() -> None:
    X, Y = _data()
    columns = [f"feature_{index}" for index in range(X.shape[1])]
    X_frame = pd.DataFrame(X, columns=columns)

    model = _fixed_estimator().fit(X_frame, Y)
    np.testing.assert_array_equal(model.feature_names_in_, columns)
    np.testing.assert_array_equal(
        model.get_feature_names_out(),
        ["piplsregression0", "piplsregression1"],
    )
    transformed = model.set_output(transform="pandas").transform(X_frame)
    assert list(transformed.columns) == ["piplsregression0", "piplsregression1"]


def test_path_and_regression_selected_outputs_are_easy_to_switch() -> None:
    X, Y = _data()
    base = _fixed_estimator()
    direct = clone(base).fit(X, Y)
    search = PiPLSSearchCV(
        estimator=base,
        n_components_values=[2],
        predictor_rank_values=[3],
        max_predictor_rank=3,
        cv=3,
        n_jobs=1,
    ).fit(X, Y)
    model = search.refit(X, Y, n_components=2)

    best = search.select(rule="best_score")
    assert best.n_components == 2
    assert best.predictor_rank == 3
    selected = search.select(n_components=2)
    assert selected.predictor_rank == 3
    assert isinstance(model, PiPLSRegression)
    np.testing.assert_allclose(model.predict(X), direct.predict(X))
    np.testing.assert_allclose(model.coef_, direct.coef_)
    np.testing.assert_allclose(
        model.decomposition_.dilation,
        direct.decomposition_.dilation,
    )
    x_model, y_model = model.transform(X, Y)
    x_direct, y_direct = direct.transform(X, Y)
    np.testing.assert_allclose(x_model, x_direct)
    np.testing.assert_allclose(y_model, y_direct)


def test_refit_returns_pipeline_without_flattening_coefficients() -> None:
    X, Y = _data()
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("regression", _fixed_estimator()),
        ]
    )
    search = PiPLSSearchCV(
        estimator=pipeline,
        n_components_values=[2],
        predictor_rank_values=[3],
        max_predictor_rank=3,
        cv=3,
        n_jobs=1,
    ).fit(X, Y)
    model = search.refit(X, Y, n_components=2)

    assert isinstance(model, Pipeline)
    assert isinstance(model.named_steps["regression"], PiPLSRegression)
    assert model.selection_ == search.select(n_components=2)
    assert not hasattr(model.named_steps["regression"], "selection_")
    assert not hasattr(search, "coef_")


def test_pipeline_can_preserve_external_predictor_scaling_and_scale_responses() -> None:
    X, Y = _data()
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "regression",
                PiPLSRegression(
                    n_components=2,
                    predictor_rank=3,
                    scale_x=False,
                    scale_y=True,
                    svd_solver="full",
                ),
            ),
        ]
    )
    search = PiPLSSearchCV(
        estimator=pipeline,
        n_components_values=[2],
        predictor_rank_values=[3],
        max_predictor_rank=3,
        cv=3,
        n_jobs=1,
    ).fit(X, Y)
    model = search.refit(X, Y, n_components=2)
    regression = model.named_steps["regression"]

    np.testing.assert_allclose(regression.x_scale_, 1.0)
    np.testing.assert_allclose(regression.y_scale_, np.std(Y, axis=0, ddof=1))


def test_minimum_cv_mse_tolerance_refit_preserves_pipeline_composition() -> None:
    X, Y = _data()
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("regression", _fixed_estimator()),
        ]
    )
    search = PiPLSSearchCV(
        estimator=pipeline,
        n_components_values=[1, 2, 3],
        predictor_rank_values=[1, 2, 3, 4],
        search_method="exhaustive",
        cv=3,
        n_jobs=1,
    ).fit(X, Y)
    expected = search.select(
        rule="minimum_cv_mse",
        relative_tolerance=0.10,
        absolute_tolerance=0.05,
    )

    model = search.refit(X, Y, selection=expected)

    assert isinstance(model, Pipeline)
    selected_pipls = model.named_steps["regression"]
    assert selected_pipls.n_components == expected.n_components
    assert selected_pipls.predictor_rank == expected.predictor_rank
    assert isinstance(model.selection_, PiPLSSelection)
    assert model.selection_ is expected
    assert model.selection_.reference_minimum == expected.reference_minimum
    assert model.selection_.cv_mse_threshold == expected.cv_mse_threshold


def test_refitted_model_score_accepts_sample_weight_like_regression() -> None:
    X, Y = _data()
    search = PiPLSSearchCV(
        estimator=_fixed_estimator(),
        n_components_values=[2],
        predictor_rank_values=[3],
        max_predictor_rank=3,
        cv=3,
        n_jobs=1,
    ).fit(X, Y)
    model = search.refit(X, Y, n_components=2)
    weights = np.linspace(1.0, 2.0, X.shape[0])

    assert model.score(X, Y, sample_weight=weights) == pytest.approx(
        _fixed_estimator().fit(X, Y).score(X, Y, sample_weight=weights)
    )


def test_refit_preserves_estimator_output_configuration() -> None:
    X, Y = _data()
    columns = [f"feature_{index}" for index in range(X.shape[1])]
    X_frame = pd.DataFrame(X, columns=columns)
    template = _fixed_estimator().set_output(transform="pandas")
    search = PiPLSSearchCV(
        estimator=template,
        n_components_values=[2],
        predictor_rank_values=[3],
        max_predictor_rank=3,
        cv=3,
        n_jobs=1,
    ).fit(X_frame, Y)
    model = search.refit(X_frame, Y, n_components=2)

    np.testing.assert_array_equal(search.feature_names_in_, columns)
    np.testing.assert_array_equal(model.feature_names_in_, columns)
    transformed = model.transform(X_frame)
    assert list(transformed.columns) == ["piplsregression0", "piplsregression1"]


def test_path_preserves_dataframe_columns_inside_pipeline_folds() -> None:
    X, Y = _data()
    columns = [f"feature_{index}" for index in range(X.shape[1])]
    X_frame = pd.DataFrame(X, columns=columns)
    selected = columns[:5]
    pipeline = Pipeline(
        [
            (
                "columns",
                ColumnTransformer(
                    [("selected", StandardScaler(), selected)],
                    remainder="drop",
                ),
            ),
            (
                "regression",
                PiPLSRegression(
                    n_components=2,
                    predictor_rank=3,
                    scale=False,
                    svd_solver="full",
                    random_state=None,
                ),
            ),
        ]
    )
    search = PiPLSSearchCV(
        estimator=pipeline,
        n_components_values=[2],
        predictor_rank_values=[3],
        max_predictor_rank=3,
        cv=3,
        n_jobs=1,
    ).fit(X_frame, Y)
    model = search.refit(X_frame, Y, n_components=2)

    assert isinstance(model, Pipeline)
    selected_pipls = model.named_steps["regression"]
    assert selected_pipls.n_features_in_ == len(selected)
    assert model.predict(X_frame).shape == Y.shape
    assert np.isfinite(search.scorer_(model, X_frame, Y))


@pytest.mark.parametrize(
    "estimator",
    [
        PiPLSRegression(
            n_components=1,
            predictor_rank=1,
            scale=False,
            svd_solver="full",
            random_state=None,
        ),
    ],
)
def test_sklearn_common_checks_except_cross_decomposition_tuple_contract(
    estimator: Any,
) -> None:
    parameters = inspect.signature(check_estimator).parameters
    if "expected_failed_checks" not in parameters:
        pytest.skip("This scikit-learn release cannot declare expected cross-decomposition checks.")
    reason = (
        "Pi-PLS mirrors PLSRegression: fit_transform(X, y) and transform(X, y) "
        "return predictor and response scores as a tuple. Generic transformer checks "
        "special-case sklearn cross-decomposition class names only."
    )
    check_estimator(
        estimator,
        expected_failed_checks={
            "check_transformer_data_not_an_array": reason,
            "check_transformer_general": reason,
        },
        on_skip=None,
    )


def test_inverse_transform_matches_documented_least_squares_reconstruction() -> None:
    X, Y = _data()
    model = _fixed_estimator().fit(X, Y)

    x_scores, y_scores = model.transform(X, Y)
    X_reconstructed, Y_reconstructed = model.inverse_transform(x_scores, y_scores)

    expected_X = (x_scores @ model.x_loadings_.T) * model.x_scale_ + model.x_mean_
    expected_Y = (y_scores @ model.y_loadings_.T) * model.y_scale_ + model.y_mean_
    np.testing.assert_allclose(X_reconstructed, expected_X)
    np.testing.assert_allclose(Y_reconstructed, expected_Y)


def test_decomposition_is_single_source_of_truth_for_pipls_specific_arrays() -> None:
    X, Y = _data()
    model = _fixed_estimator().fit(X, Y)

    assert model.x_rotations_ is model.decomposition_.predictor_directions
    assert model.y_rotations_ is model.decomposition_.response_directions
    assert not model.x_rotations_.flags.writeable


def test_path_restricts_estimator_scope_to_direct_or_final_pipeline_pipls() -> None:
    X, Y = _data()
    unsupported = TransformedTargetRegressor(regressor=_fixed_estimator())

    with pytest.raises(ValueError, match="PiPLSRegression or a sklearn Pipeline"):
        PiPLSSearchCV(estimator=unsupported).fit(X, Y)

    invalid_pipeline = Pipeline([("pipls", _fixed_estimator()), ("scale", StandardScaler())])
    with pytest.raises(ValueError, match="pipelines must end"):
        PiPLSSearchCV(estimator=invalid_pipeline).fit(X, Y)


def test_path_search_diagnostics_and_refitted_model_are_sklearn_like() -> None:
    X, Y = _data()
    search = PiPLSSearchCV(
        estimator=_fixed_estimator(),
        n_components_values=[2],
        predictor_rank_values=[3],
        max_predictor_rank=3,
        cv=None,
        scoring=None,
        n_jobs=1,
    ).fit(X, Y)
    model = search.refit(X, Y, n_components=2)

    assert search.n_splits_ == 5
    assert callable(search.scorer_)
    assert {
        "mean_fit_time",
        "std_fit_time",
        "mean_score_time",
        "std_score_time",
    } <= search.cv_results_.keys()
    assert search.component_path_.predictor_rank_policy == "fixed"
    assert isinstance(search.component_path_, PiPLSComponentPath)
    assert search.select(n_components=2).predictor_rank == 3
    x_scores, y_scores = model.transform(X, Y)
    X_reconstructed, Y_reconstructed = model.inverse_transform(x_scores, y_scores)
    direct_X, direct_Y = _fixed_estimator().fit(X, Y).inverse_transform(
        x_scores,
        y_scores,
    )
    np.testing.assert_allclose(X_reconstructed, direct_X)
    np.testing.assert_allclose(Y_reconstructed, direct_Y)


def test_path_cv_results_use_pipeline_independent_candidate_columns() -> None:
    X, Y = _data()
    estimator = Pipeline(
        [
            ("scale", StandardScaler()),
            ("pipls", _fixed_estimator()),
        ]
    )
    search = PiPLSSearchCV(
        estimator=estimator,
        n_components_values=[1, 2],
        predictor_rank_values=[1, 2, 3],
        max_predictor_rank=3,
        search_method="exhaustive",
        cv=3,
        n_jobs=1,
    ).fit(X, Y)

    expected_columns = {
        "n_components",
        "predictor_rank",
        "mean_test_score",
        "std_test_score",
        "rank_test_score",
        "mean_fit_time",
        "std_fit_time",
        "mean_score_time",
        "std_score_time",
        "mean_response_standardized_mse",
        "std_response_standardized_mse",
    }
    for split_index in range(search.n_splits_):
        expected_columns.add(f"split{split_index}_test_score")
        expected_columns.add(
            f"split{split_index}_response_standardized_mse"
        )

    assert set(search.cv_results_) == expected_columns
    candidate_count = search.cv_results_["n_components"].size
    assert all(
        isinstance(values, np.ndarray) and values.shape == (candidate_count,)
        for values in search.cv_results_.values()
    )


def test_refitted_pipeline_exposes_only_pipeline_supported_transformations() -> None:
    X, Y = _data()
    pipeline = Pipeline(
        [
            (
                "columns",
                ColumnTransformer(
                    [("selected", StandardScaler(), [0, 1, 2, 3])],
                    remainder="drop",
                ),
            ),
            ("regression", _fixed_estimator()),
        ]
    )
    search = PiPLSSearchCV(
        estimator=pipeline,
        n_components_values=[2],
        predictor_rank_values=[3],
        max_predictor_rank=3,
        cv=3,
    ).fit(X, Y)
    model = search.refit(X, Y, n_components=2)

    assert hasattr(model, "transform")
    assert not hasattr(model, "inverse_transform")
    assert not hasattr(search, "transform")
    assert not hasattr(search, "inverse_transform")
