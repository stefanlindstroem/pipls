"""Example-local rendering helpers for the Pulp score-loading biplot."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from matplotlib.patches import FancyArrowPatch
from numpy.typing import NDArray

from pipls.inspection import BiplotCoordinates

try:
    import textalloc as _textalloc
except ModuleNotFoundError as exc:
    if exc.name != "textalloc":
        raise
    _textalloc = None

FloatArray = NDArray[np.float64]

PREDICTOR_LABEL_FONTSIZE = 9
TEXTALLOC_MIN_DISTANCE = 0.01125
TEXTALLOC_MAX_DISTANCE = 0.15


def _validate_predictor_names(
    predictor_names: Sequence[str],
    predictor_xy: FloatArray,
) -> tuple[str, ...]:
    names = tuple(str(name) for name in predictor_names)
    if len(names) != predictor_xy.shape[0]:
        raise ValueError(
            "predictor_names must contain one label for each predictor endpoint."
        )
    return names


def _arrow_segments(endpoints: FloatArray) -> tuple[list[FloatArray], list[FloatArray]]:
    x_lines = [np.array([0.0, endpoint[0]], dtype=np.float64) for endpoint in endpoints]
    y_lines = [np.array([0.0, endpoint[1]], dtype=np.float64) for endpoint in endpoints]
    return x_lines, y_lines


def _draw_biplot_geometry(
    axis: Any,
    biplot: BiplotCoordinates,
    *,
    title: str | None,
) -> FloatArray:
    sample_xy = np.asarray(biplot.sample_coordinates, dtype=np.float64)
    predictor_xy = np.asarray(biplot.predictor_coordinates, dtype=np.float64)
    axis.scatter(
        sample_xy[:, 0],
        sample_xy[:, 1],
        alpha=0.75,
        label="Samples",
    )
    for endpoint in predictor_xy:
        axis.add_patch(
            FancyArrowPatch(
                (0.0, 0.0),
                (float(endpoint[0]), float(endpoint[1])),
                arrowstyle="->",
                mutation_scale=10.0,
                linewidth=1.0,
            )
        )
    first, second = (int(value) for value in biplot.component_indices)
    axis.axhline(0.0, linewidth=0.8, linestyle="--", color="0.45")
    axis.axvline(0.0, linewidth=0.8, linestyle="--", color="0.45")
    axis.set_xlabel(f"Balanced component {first + 1}")
    axis.set_ylabel(f"Balanced component {second + 1}")
    if title is not None:
        axis.set_title(title)
    axis.set_aspect("equal", adjustable="datalim")
    axis.margins(0.1)
    axis.legend()
    return predictor_xy


def plot_pulp_biplot_simple(
    axis: Any,
    biplot: BiplotCoordinates,
    *,
    predictor_names: Sequence[str],
    title: str | None = None,
) -> tuple[Any, ...]:
    """Render the Pulp biplot with labels fixed at predictor endpoints."""
    predictor_xy = _draw_biplot_geometry(axis, biplot, title=title)
    names = _validate_predictor_names(predictor_names, predictor_xy)
    return tuple(
        axis.text(
            float(endpoint[0]),
            float(endpoint[1]),
            name,
            fontsize=PREDICTOR_LABEL_FONTSIZE,
        )
        for endpoint, name in zip(predictor_xy, names, strict=True)
    )


def plot_pulp_biplot_textalloc(
    axis: Any,
    biplot: BiplotCoordinates,
    *,
    predictor_names: Sequence[str],
    title: str | None = None,
) -> tuple[Any, ...]:
    """Render the Pulp biplot with line-aware predictor-label allocation."""
    if _textalloc is None:
        raise RuntimeError("textalloc is unavailable; use plot_pulp_biplot_simple instead.")

    predictor_xy = _draw_biplot_geometry(axis, biplot, title=title)
    names = _validate_predictor_names(predictor_names, predictor_xy)
    x_lines, y_lines = _arrow_segments(predictor_xy)
    result = _textalloc.allocate(
        axis,
        predictor_xy[:, 0],
        predictor_xy[:, 1],
        names,
        x_lines=x_lines,
        y_lines=y_lines,
        textsize=PREDICTOR_LABEL_FONTSIZE,
        min_distance=TEXTALLOC_MIN_DISTANCE,
        max_distance=TEXTALLOC_MAX_DISTANCE,
        draw_all=True,
        draw_lines=False,
    )
    return tuple(result[2])


def plot_pulp_biplot(
    axis: Any,
    biplot: BiplotCoordinates,
    *,
    predictor_names: Sequence[str],
    title: str | None = None,
) -> tuple[Any, ...]:
    """Render the preferred Pulp biplot with a plain-Matplotlib fallback."""
    if _textalloc is None:
        return plot_pulp_biplot_simple(
            axis,
            biplot,
            predictor_names=predictor_names,
            title=title,
        )
    return plot_pulp_biplot_textalloc(
        axis,
        biplot,
        predictor_names=predictor_names,
        title=title,
    )
