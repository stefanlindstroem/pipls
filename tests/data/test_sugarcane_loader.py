from __future__ import annotations

from collections.abc import Mapping

from pipls.datasets import load_sugarcane

FEATURE_NAMES = tuple(str(wavelength) for wavelength in range(780, 2501))
TARGET_NAMES = ("TS", "CP", "ADF", "IVOMD")


def test_load_sugarcane_names_provenance_and_sample_alignment() -> None:
    dataset = load_sugarcane()

    assert dataset.feature_names == FEATURE_NAMES
    assert dataset.target_names == TARGET_NAMES

    provenance = dataset.metadata["provenance"]
    assert isinstance(provenance, Mapping)
    assert provenance["source"].endswith("10.17632/mjttsjfj2s.1")
    assert provenance["license"] == "CC-BY-4.0"
    assert "Chaix" in provenance["citation"]
    assert provenance["version"] == "1"

    sample_alignment = dataset.metadata["sample_alignment"]
    assert isinstance(sample_alignment, Mapping)
    assert sample_alignment["source_sample_ids"] == [
        sample for sample in range(100, 160) if sample not in {103, 105, 111}
    ]
    assert sample_alignment["excluded_source_sample_ids"] == [103, 105, 111]
