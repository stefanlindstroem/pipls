from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest

from pipls.datasets import load_tobacco

TARGET_NAMES = (
    "Total Alkaloids",
    "Reducing Sugars",
    "Total Sugars",
    "Total Nitrogen",
    "K",
    "cl",
    "pH",
    "Starch",
    "Neochlorogenic Acid",
    "Chlorogenic Acid",
    "Cryptochlorogenic Acid",
    "Scopoletin",
    "Rutin",
)


def test_load_tobacco_names_provenance_and_spectral_axis() -> None:
    dataset = load_tobacco()
    wavenumbers = np.asarray(dataset.feature_names, dtype=np.float64)

    assert dataset.feature_names[0] == "10001.0283203125"
    assert dataset.feature_names[-1] == "3999.63989257813"
    assert np.all(np.diff(wavenumbers) < 0.0)
    assert dataset.target_names == TARGET_NAMES

    provenance = dataset.metadata["provenance"]
    assert isinstance(provenance, Mapping)
    assert provenance["source"].endswith("10.17632/9z7dgdtggk.1")
    assert provenance["license"] == "CC-BY-4.0"
    assert "Chen" in provenance["citation"]
    assert provenance["version"] == "1"

    variables = dataset.metadata["variables"]
    assert isinstance(variables, Mapping)
    predictors = variables["predictors"]
    assert isinstance(predictors, Mapping)
    axis = predictors["axis"]
    assert isinstance(axis, Mapping)
    assert axis["name"] == "wavenumber"
    assert axis["unit"] == "cm^-1"
    assert axis["ordering"] == "decreasing"
    assert axis["step"] == pytest.approx(-3.856933436849431)
