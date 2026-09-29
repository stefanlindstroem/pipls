from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping

import numpy as np
import pytest

from pipls.datasets import PiPLSDataset, load_pulp, load_sugarcane, load_tobacco


@pytest.mark.parametrize(
    ("loader", "x_shape", "y_shape"),
    [
        (load_pulp, (46, 14), (46, 8)),
        (load_sugarcane, (57, 1721), (57, 4)),
        (load_tobacco, (347, 1557), (347, 13)),
    ],
)
def test_reference_loader_contract(
    loader: Callable[..., object],
    x_shape: tuple[int, int],
    y_shape: tuple[int, int],
) -> None:
    return_parameter = inspect.signature(loader).parameters["return_X_y"]
    assert return_parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert return_parameter.default is False

    with pytest.raises(TypeError, match="return_X_y must be a boolean"):
        loader(return_X_y=1)

    dataset = loader()
    second_dataset = loader()
    assert isinstance(dataset, PiPLSDataset)
    assert isinstance(second_dataset, PiPLSDataset)

    assert dataset.X.shape == x_shape
    assert dataset.Y.shape == y_shape
    assert dataset.X.dtype == np.float64
    assert dataset.Y.dtype == np.float64
    assert not dataset.X.flags.writeable
    assert not dataset.Y.flags.writeable
    assert np.isfinite(dataset.X).all()
    assert np.isfinite(dataset.Y).all()
    assert not np.shares_memory(dataset.X, second_dataset.X)
    assert not np.shares_memory(dataset.Y, second_dataset.Y)

    arrays = loader(return_X_y=True)
    second_arrays = loader(return_X_y=True)
    assert isinstance(arrays, tuple)
    assert isinstance(second_arrays, tuple)
    X, Y = arrays
    second_X, second_Y = second_arrays
    np.testing.assert_array_equal(X, dataset.X)
    np.testing.assert_array_equal(Y, dataset.Y)
    assert not X.flags.writeable
    assert not Y.flags.writeable
    assert not np.shares_memory(X, second_X)
    assert not np.shares_memory(Y, second_Y)

    with pytest.raises(ValueError):
        X[0, 0] = 0.0
    with pytest.raises(ValueError):
        Y[0, 0] = 0.0

    assert isinstance(dataset.metadata, Mapping)
    assert dataset.metadata["schema_version"] == 1
    assert dataset.metadata["feature_names"] == list(dataset.feature_names)
    assert dataset.metadata["target_names"] == list(dataset.target_names)
    assert dataset.metadata["dimensions"] == {
        "n_samples": x_shape[0],
        "n_features": x_shape[1],
        "n_targets": y_shape[1],
    }
