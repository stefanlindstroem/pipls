from __future__ import annotations

from collections.abc import Mapping

from pipls.datasets import load_pulp

FEATURE_NAMES = (
    "Shives",
    "Fines B",
    "L (arith)",
    "L (lw)",
    "L (llw)",
    "W (arith)",
    "W (lw)",
    "W (llw)",
    "C (arith)",
    "C (lw)",
    "C (llw)",
    "F (arith)",
    "F (lw)",
    "F (llw)",
)
TARGET_NAMES = (
    "CSF",
    "Density",
    "TI",
    "Elongation",
    "TEA",
    "TSI",
    "Tear index",
    "s",
)


def test_load_pulp_names_and_provenance() -> None:
    dataset = load_pulp()

    assert dataset.feature_names == FEATURE_NAMES
    assert dataset.target_names == TARGET_NAMES

    provenance = dataset.metadata["provenance"]
    assert isinstance(provenance, Mapping)
    assert provenance["source"].endswith("10.1016/j.compchemeng.2025.109143")
    assert provenance["license"] == "CC-BY-4.0"
    assert "Lindström" in provenance["citation"]
    assert provenance["version"] == "1"
