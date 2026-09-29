from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pipls.inspection import BiplotCoordinates


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_pulp_biplot() -> ModuleType:
    path = _repository_root() / "examples" / "_support" / "pulp_biplot.py"
    spec = importlib.util.spec_from_file_location("pulp_biplot_example", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load example module from {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PULP_BIPLOT = _load_pulp_biplot()


@pytest.fixture
def coordinates() -> BiplotCoordinates:
    return BiplotCoordinates(
        sample_coordinates=np.array([[0.1, 0.2], [-0.3, 0.4], [0.2, -0.1]]),
        predictor_coordinates=np.array([[1.5, -2.0], [-0.25, 4.0]]),
        component_indices=np.array([0, 1]),
        scaling_factors=np.array([1.0, 1.0]),
    )


def test_plain_biplot_returns_predictor_labels_without_textalloc(
    monkeypatch: pytest.MonkeyPatch,
    coordinates: BiplotCoordinates,
) -> None:
    monkeypatch.setattr(PULP_BIPLOT, "_textalloc", None)
    figure, axis = plt.subplots()
    try:
        labels = PULP_BIPLOT.plot_pulp_biplot(
            axis,
            coordinates,
            predictor_names=("A", "B"),
        )

        assert tuple(label.get_text() for label in labels) == ("A", "B")
        assert all(np.isfinite(label.get_position()).all() for label in labels)
    finally:
        plt.close(figure)


def test_textalloc_biplot_supplies_predictor_arrow_obstacles(
    monkeypatch: pytest.MonkeyPatch,
    coordinates: BiplotCoordinates,
) -> None:
    captured: dict[str, object] = {}
    text_objects = [object(), object()]

    def allocate(*args: object, **kwargs: object) -> tuple[object, object, list[object], object]:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return object(), object(), text_objects, object()

    monkeypatch.setattr(PULP_BIPLOT, "_textalloc", SimpleNamespace(allocate=allocate))
    figure, axis = plt.subplots()
    try:
        result = PULP_BIPLOT.plot_pulp_biplot(
            axis,
            coordinates,
            predictor_names=("A", "B"),
        )

        assert result == tuple(text_objects)
        args = captured["args"]
        kwargs = captured["kwargs"]
        assert args[0] is axis
        np.testing.assert_array_equal(args[1], coordinates.predictor_coordinates[:, 0])
        np.testing.assert_array_equal(args[2], coordinates.predictor_coordinates[:, 1])
        assert args[3] == ("A", "B")
        assert "x_scatter" not in kwargs
        assert "y_scatter" not in kwargs
        x_lines = kwargs["x_lines"]
        y_lines = kwargs["y_lines"]
        assert len(x_lines) == len(y_lines) == coordinates.predictor_coordinates.shape[0]
        for index, endpoint in enumerate(coordinates.predictor_coordinates):
            np.testing.assert_array_equal(x_lines[index], np.array([0.0, endpoint[0]]))
            np.testing.assert_array_equal(y_lines[index], np.array([0.0, endpoint[1]]))
    finally:
        plt.close(figure)


def test_explicit_textalloc_renderer_requires_textalloc(
    monkeypatch: pytest.MonkeyPatch,
    coordinates: BiplotCoordinates,
) -> None:
    monkeypatch.setattr(PULP_BIPLOT, "_textalloc", None)
    figure, axis = plt.subplots()
    try:
        with pytest.raises(RuntimeError, match="textalloc is unavailable"):
            PULP_BIPLOT.plot_pulp_biplot_textalloc(
                axis,
                coordinates,
                predictor_names=("A", "B"),
            )
    finally:
        plt.close(figure)


def test_biplot_validates_predictor_label_alignment(
    monkeypatch: pytest.MonkeyPatch,
    coordinates: BiplotCoordinates,
) -> None:
    monkeypatch.setattr(PULP_BIPLOT, "_textalloc", None)
    figure, axis = plt.subplots()
    try:
        with pytest.raises(ValueError, match="one label"):
            PULP_BIPLOT.plot_pulp_biplot(
                axis,
                coordinates,
                predictor_names=("A",),
            )
    finally:
        plt.close(figure)
