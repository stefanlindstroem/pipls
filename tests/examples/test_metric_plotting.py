from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_metric_plotting() -> ModuleType:
    path = _repository_root() / "examples" / "_support" / "metric_plotting.py"
    spec = importlib.util.spec_from_file_location("metric_plotting_example", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load example module from {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


METRIC_PLOTTING = _load_metric_plotting()


def test_response_r2_ylim_uses_zero_to_one_for_nonnegative_values() -> None:
    assert METRIC_PLOTTING.response_r2_ylim([0.2, 0.7, 1.0]) == (0.0, 1.0)


def test_response_r2_ylim_exposes_negative_values_with_downward_padding() -> None:
    lower, upper = METRIC_PLOTTING.response_r2_ylim([-0.4, 0.2, 0.8])

    assert lower < -0.4
    assert lower <= 0.0
    assert upper == 1.0
    assert lower == pytest.approx(-0.47)


def test_response_r2_ylim_can_disable_negative_padding() -> None:
    assert METRIC_PLOTTING.response_r2_ylim(
        np.array([-0.25, 0.5]),
        negative_padding_fraction=0.0,
    ) == (-0.25, 1.0)


@pytest.mark.parametrize(
    ("values", "padding"),
    [
        ([], 0.05),
        ([0.2, np.nan], 0.05),
        ([0.2], -0.1),
        ([0.2], np.inf),
    ],
)
def test_response_r2_ylim_rejects_invalid_inputs(
    values: list[float],
    padding: float,
) -> None:
    with pytest.raises(ValueError):
        METRIC_PLOTTING.response_r2_ylim(
            values,
            negative_padding_fraction=padding,
        )
