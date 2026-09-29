from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest

from pipls.datasets import PiPLSDataset


def _dataset(**overrides: object) -> PiPLSDataset:
    values: dict[str, object] = {
        "X": np.arange(12, dtype=np.float64).reshape(4, 3),
        "Y": np.arange(8, dtype=np.float64).reshape(4, 2),
        "feature_names": ("a", "b", "c"),
        "target_names": ("u", "v"),
        "metadata": {"nested": {"values": [1, 2]}},
    }
    values.update(overrides)
    return PiPLSDataset(**values)  # type: ignore[arg-type]


def test_dataset_normalizes_single_target_and_exposes_canonical_dimensions() -> None:
    dataset = _dataset(
        Y=np.arange(4, dtype=np.float64),
        target_names=("response",),
    )

    assert dataset.X.shape == (4, 3)
    assert dataset.Y.shape == (4, 1)
    assert dataset.n_samples == 4
    assert dataset.n_features == 3
    assert dataset.n_targets == 1


def test_dataset_copies_arrays_and_top_level_metadata() -> None:
    X = np.arange(12, dtype=np.float64).reshape(4, 3)
    metadata = {"instrument": "example"}
    dataset = _dataset(X=X, metadata=metadata)

    X[0, 0] = -999.0
    metadata["instrument"] = "changed"
    metadata["new"] = 1

    assert dataset.X[0, 0] == 0.0
    assert not dataset.X.flags.writeable
    assert not dataset.Y.flags.writeable
    assert isinstance(dataset.metadata, Mapping)
    assert dataset.metadata == {"instrument": "example"}


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("X", [["a"]], TypeError),
        ("X", np.ones(4), ValueError),
        ("X", np.array([[np.nan], [1.0], [2.0], [3.0]]), ValueError),
        ("Y", np.ones((3, 2)), ValueError),
        ("feature_names", ("a", "b"), ValueError),
        ("feature_names", ("a", "a", "c"), ValueError),
        ("target_names", ("u", ""), TypeError),
        ("metadata", (), TypeError),
    ],
)
def test_dataset_rejects_invalid_public_inputs(
    field: str,
    value: object,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        _dataset(**{field: value})
