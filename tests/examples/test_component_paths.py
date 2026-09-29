from __future__ import annotations

import importlib.util
import pickle
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_example_module(relative_path: str, module_name: str) -> ModuleType:
    path = _repository_root() / "examples" / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load example module from {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PLS_PATH = _load_example_module("_support/pls_component_path.py", "pls_component_path_example")


def test_standard_pls_path_is_deterministic_and_immutable() -> None:
    rng = np.random.default_rng(1729)
    X = pd.DataFrame(rng.normal(size=(30, 6)))
    Y = pd.DataFrame(rng.normal(size=(30, 3)))

    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    first = PLS_PATH.evaluate_pls_component_path(
        X, Y, max_n_components=3, cv=cv
    )
    second = PLS_PATH.evaluate_pls_component_path(
        X, Y, max_n_components=3, cv=cv
    )
    arrays = PLS_PATH.evaluate_pls_component_path(
        X.to_numpy(), Y.to_numpy(), max_n_components=3, cv=cv
    )

    np.testing.assert_array_equal(first.n_components, second.n_components)
    np.testing.assert_array_equal(first.n_components, arrays.n_components)
    np.testing.assert_allclose(first.cv_mse_mean, second.cv_mse_mean)
    np.testing.assert_allclose(first.cv_mse_std, second.cv_mse_std)
    np.testing.assert_allclose(first.cv_mse_mean, arrays.cv_mse_mean)
    np.testing.assert_allclose(first.cv_mse_std, arrays.cv_mse_std)
    assert first.n_components.tolist() == [1, 2, 3]
    assert first.algorithm == "NIPALS"
    assert first.n_splits == 5
    assert first.n_components.dtype == np.dtype(np.intp)
    assert first.cv_mse_mean.dtype == np.dtype(np.float64)
    assert first.cv_mse_std.dtype == np.dtype(np.float64)
    assert np.isfinite(first.cv_mse_mean).all()
    assert (first.cv_mse_std >= 0.0).all()
    assert not first.n_components.flags.writeable
    assert not first.cv_mse_mean.flags.writeable
    assert not first.cv_mse_std.flags.writeable


def test_standard_pls_path_accepts_materialized_shared_splits() -> None:
    rng = np.random.default_rng(1730)
    X = pd.DataFrame(rng.normal(size=(30, 6)))
    Y = pd.DataFrame(rng.normal(size=(30, 3)))

    splitter = KFold(n_splits=5, shuffle=True, random_state=0)
    splits = list(splitter.split(X, Y))
    from_splitter = PLS_PATH.evaluate_pls_component_path(
        X, Y, max_n_components=3, cv=splitter
    )
    from_materialized = PLS_PATH.evaluate_pls_component_path(
        X, Y, max_n_components=3, cv=splits
    )

    np.testing.assert_array_equal(
        from_materialized.n_components,
        from_splitter.n_components,
    )
    np.testing.assert_allclose(
        from_materialized.cv_mse_mean,
        from_splitter.cv_mse_mean,
    )
    np.testing.assert_allclose(
        from_materialized.cv_mse_std,
        from_splitter.cv_mse_std,
    )
    assert from_materialized.n_splits == len(splits) == 5


def test_standard_pls_path_defensively_copies_and_pickles() -> None:
    components = np.array([1, 2, 3])
    means = np.array([0.9, 0.6, 0.55])
    fold_sd = np.array([0.12, 0.09, 0.08])
    path = PLS_PATH.PLSComponentPath(
        n_components=components,
        cv_mse_mean=means,
        cv_mse_std=fold_sd,
        algorithm="NIPALS",
        n_splits=5,
    )

    components[0] = 7
    means[0] = 7.0
    fold_sd[0] = 7.0
    assert path.n_components.tolist() == [1, 2, 3]
    assert path.cv_mse_mean.tolist() == [0.9, 0.6, 0.55]
    assert path.cv_mse_std.tolist() == [0.12, 0.09, 0.08]

    restored = pickle.loads(pickle.dumps(path))
    np.testing.assert_array_equal(restored.n_components, path.n_components)
    np.testing.assert_allclose(restored.cv_mse_mean, path.cv_mse_mean)
    np.testing.assert_allclose(restored.cv_mse_std, path.cv_mse_std)
    assert not restored.n_components.flags.writeable
    assert not restored.cv_mse_mean.flags.writeable
    assert not restored.cv_mse_std.flags.writeable


def test_standard_pls_path_rejects_invalid_arrays() -> None:
    with pytest.raises(ValueError, match="same length"):
        PLS_PATH.PLSComponentPath(
            n_components=[1, 2],
            cv_mse_mean=[0.9],
            cv_mse_std=[0.1, 0.1],
            algorithm="NIPALS",
            n_splits=5,
        )
    with pytest.raises(ValueError, match="strictly ascending"):
        PLS_PATH.PLSComponentPath(
            n_components=[1, 1],
            cv_mse_mean=[0.9, 0.8],
            cv_mse_std=[0.1, 0.1],
            algorithm="NIPALS",
            n_splits=5,
        )
    with pytest.raises(ValueError, match="nonnegative"):
        PLS_PATH.PLSComponentPath(
            n_components=[1, 2],
            cv_mse_mean=[0.9, 0.8],
            cv_mse_std=[0.1, -0.1],
            algorithm="NIPALS",
            n_splits=5,
        )
    PLS_PATH.PLSComponentPath(
        n_components=[1, 2],
        cv_mse_mean=[0.9, 0.8],
        cv_mse_std=[0.0, 0.0],
        algorithm="NIPALS",
        n_splits=1,
    )


def test_nested_pls_path_matches_separate_pls_fits() -> None:
    rng = np.random.default_rng(2718)
    X = pd.DataFrame(rng.normal(size=(30, 6)))
    Y = pd.DataFrame(rng.normal(size=(30, 3)))
    splitter = KFold(n_splits=5, shuffle=True, random_state=0)
    path = PLS_PATH.evaluate_pls_component_path(
        X, Y, max_n_components=3, cv=splitter
    )

    for n_components in range(1, 4):
        split_mse: list[float] = []
        for train, validation in splitter.split(X):
            model = PLSRegression(n_components=n_components, scale=True)
            model.fit(X.iloc[train], Y.iloc[train])
            prediction = model.predict(X.iloc[validation])
            response_scale = Y.iloc[train].std(axis=0, ddof=1).to_numpy()
            residual = Y.iloc[validation].to_numpy() - prediction
            split_mse.append(float(np.mean((residual / response_scale) ** 2)))

        index = n_components - 1
        assert path.cv_mse_mean[index] == pytest.approx(np.mean(split_mse))
        assert path.cv_mse_std[index] == pytest.approx(np.std(split_mse))
