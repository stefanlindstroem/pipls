from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Callable
from importlib import resources
from typing import Any

import numpy as np

from pipls.datasets import PiPLSDataset, load_pulp, load_sugarcane, load_tobacco

_DATASET_LOADERS: dict[str, Callable[[], PiPLSDataset]] = {
    "pulp": load_pulp,
    "sugarcane": load_sugarcane,
    "tobacco": load_tobacco,
}


def _resource_root(dataset_id: str):
    return resources.files("pipls").joinpath("_data").joinpath(dataset_id)


def _metadata(dataset_id: str) -> dict[str, Any]:
    loaded = json.loads(
        _resource_root(dataset_id).joinpath("metadata.json").read_text(encoding="utf-8")
    )
    assert isinstance(loaded, dict)
    return loaded


def _canonical_array_hash(array: np.ndarray) -> str:
    canonical = np.asarray(array, dtype=np.dtype("<f8"), order="C")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _csv_rows(dataset_id: str, name: str) -> list[list[str]]:
    with _resource_root(dataset_id).joinpath(name).open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        return list(csv.reader(handle))


def test_reference_dataset_resources_match_declared_integrity() -> None:
    for dataset_id, loader in _DATASET_LOADERS.items():
        metadata = _metadata(dataset_id)
        resource_root = _resource_root(dataset_id)
        assert metadata["dataset"]["id"] == dataset_id

        resource_hashes = metadata["integrity"]["resource_sha256"]
        for name, expected_hash in resource_hashes.items():
            payload = resource_root.joinpath(name).read_bytes()
            assert hashlib.sha256(payload).hexdigest() == expected_hash

        dataset = loader()
        array_hashes = metadata["integrity"]["array_sha256"]
        assert _canonical_array_hash(dataset.X) == array_hashes["data_float64_c_order"]
        assert _canonical_array_hash(dataset.Y) == array_hashes[
            "target_float64_c_order"
        ]


def test_reference_dataset_metadata_matches_matrix_layout() -> None:
    for dataset_id in _DATASET_LOADERS:
        metadata = _metadata(dataset_id)
        dimensions = metadata["dimensions"]
        assert isinstance(dimensions, dict)

        for name, names_key, columns_key in (
            ("X.csv", "feature_names", "n_features"),
            ("Y.csv", "target_names", "n_targets"),
        ):
            names = metadata[names_key]
            assert isinstance(names, list)
            assert names
            assert all(isinstance(value, str) and value for value in names)
            assert len(set(names)) == len(names)

            rows = _csv_rows(dataset_id, name)
            assert rows
            assert rows[0] == names
            assert len(rows) - 1 == dimensions["n_samples"]
            assert len(names) == dimensions[columns_key]
            assert all(len(row) == len(names) for row in rows[1:])
