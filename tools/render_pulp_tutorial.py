"""Generate deterministic single-chart figures for the Pulp tutorial."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPOSITORY_ROOT / "src"
EXAMPLES_DIR = REPOSITORY_ROOT / "examples"
for path in (SRC_DIR, EXAMPLES_DIR):
    path_entry = str(path)
    if path_entry not in sys.path:
        sys.path.insert(0, path_entry)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "pipls-pulp-tutorial"
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from _support.metric_plotting import response_r2_ylim  # noqa: E402
from _support.pulp_biplot import plot_pulp_biplot  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from sklearn.model_selection import RepeatedKFold  # noqa: E402

from pipls import PiPLSSearchCV  # noqa: E402
from pipls.component_path import (  # noqa: E402
    PiPLSComponentPath,
    PiPLSPredictorRankProfile,
    PiPLSSelection,
)
from pipls.datasets import PiPLSDataset, load_pulp  # noqa: E402
from pipls.inspection import (  # noqa: E402
    LatentStructure,
    PiPLSDisplayFactors,
    PredictionDiagnostics,
    biplot_coordinates,
    latent_structure,
    pipls_display_factors,
    prediction_diagnostics,
)
from pipls.validation import PiPLSOOFReport  # noqa: E402

DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "docs" / "assets" / "generated" / "pulp"
CV = RepeatedKFold(n_splits=5, n_repeats=10, random_state=0)
PREDICTION_KIND = "selection-conditioned OOF predictions"
DOMAIN_PREDICTOR_RELATIVE_TOLERANCE = 0.15
DOMAIN_COMPONENT_RELATIVE_TOLERANCE = 0.50
EPV_SAMPLES_PER_PREDICTOR_RANK = 5.0
FIGURE_FILENAMES = (
    "search_domain.svg",
    "conditioned_search_domain.svg",
    "conditioned_component_path.svg",
    "epv_search_domain.svg",
    "component_path.svg",
    "selected_component_path.svg",
    "predictor_rank_profile.svg",
    "biplot.svg",
    "predictor_directions.svg",
    "weighted_response_directions.svg",
    "observed_vs_predicted.svg",
    "residuals_vs_predicted.svg",
    "oof_response_r2.svg",
    "final_fit_observed_vs_predicted.svg",
    "final_fit_r2.svg",
    "final_fit_residual_distribution.svg",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _figure(*, figsize: tuple[float, float]) -> tuple[Figure, Axes]:
    return plt.subplots(figsize=figsize, layout="constrained")


def _save_svg(figure: Figure, path: Path) -> None:
    figure.savefig(
        path,
        format="svg",
        metadata={"Creator": "Pi-PLS repository", "Date": None},
    )
    plt.close(figure)


def _candidate_surface(search: PiPLSSearchCV) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_components = np.asarray(search.cv_results_["n_components"], dtype=np.intp)
    predictor_rank = np.asarray(search.cv_results_["predictor_rank"], dtype=np.intp)
    cv_mse = np.asarray(
        search.cv_results_["mean_response_standardized_mse"],
        dtype=np.float64,
    )

    component_values = np.arange(1, int(np.max(n_components)) + 1, dtype=np.intp)
    rank_values = np.arange(1, search.max_predictor_rank_ + 1, dtype=np.intp)
    surface = np.full(
        (rank_values.size, component_values.size),
        np.nan,
        dtype=np.float64,
    )
    for h_value, rank_value, mse_value in zip(
        n_components,
        predictor_rank,
        cv_mse,
        strict=True,
    ):
        surface[int(rank_value) - 1, int(h_value) - 1] = float(mse_value)
    return component_values, rank_values, surface


def _search_surface_figure(
    search: PiPLSSearchCV,
    *,
    title: str,
) -> tuple[Figure, Axes, np.ndarray, np.ndarray, np.ndarray]:
    component_values, rank_values, surface = _candidate_surface(search)
    figure, axis = _figure(figsize=(7.4, 5.0))
    colormap = plt.get_cmap("viridis").copy()
    colormap.set_bad("0.92")
    image = axis.imshow(
        np.ma.masked_invalid(surface),
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=(
            0.5,
            float(component_values[-1]) + 0.5,
            0.5,
            float(rank_values[-1]) + 0.5,
        ),
        cmap=colormap,
    )
    axis.set_xticks(component_values)
    axis.set_yticks(rank_values)
    axis.set_xticks(
        np.arange(1.5, float(component_values[-1]) + 0.5),
        minor=True,
    )
    axis.set_yticks(
        np.arange(1.5, float(rank_values[-1]) + 0.5),
        minor=True,
    )
    axis.grid(which="minor", linewidth=0.5, color="white", alpha=0.65)
    axis.tick_params(which="minor", bottom=False, left=False)
    axis.set_xlabel(r"Paired-mode count $h$")
    axis.set_ylabel(r"Predictor rank $r_\pi$")
    axis.set_title(title)
    colorbar = figure.colorbar(image, ax=axis, shrink=0.86)
    colorbar.set_label("Mean response-standardized CV-MSE")
    return figure, axis, component_values, rank_values, surface


def _render_search_domain(
    search: PiPLSSearchCV,
    *,
    output_path: Path,
) -> None:
    figure, axis, component_values, _, _ = _search_surface_figure(
        search,
        title="Pulp exhaustive search domain",
    )
    axis.text(
        float(component_values[-1]) - 1.3,
        2.1,
        r"$r_\pi < h$" + "\nnot admissible",
        ha="center",
        va="center",
    )
    _save_svg(figure, output_path)


def _relative_tolerance_path(
    search: PiPLSSearchCV,
    *,
    relative_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    component_values, rank_values, surface = _candidate_surface(search)
    exact_ranks = np.empty(component_values.size, dtype=np.intp)
    selected_ranks = np.empty(component_values.size, dtype=np.intp)
    selected_mse = np.empty(component_values.size, dtype=np.float64)

    for column, _ in enumerate(component_values):
        mse = surface[:, column]
        admissible = np.flatnonzero(np.isfinite(mse))
        admissible_mse = mse[admissible]
        exact_offset = int(np.argmin(admissible_mse))
        exact_index = int(admissible[exact_offset])
        threshold = (1.0 + relative_tolerance) * float(mse[exact_index])
        qualifying = admissible[mse[admissible] <= threshold]
        selected_index = int(qualifying[0])
        exact_ranks[column] = rank_values[exact_index]
        selected_ranks[column] = rank_values[selected_index]
        selected_mse[column] = mse[selected_index]

    return component_values, exact_ranks, selected_ranks, selected_mse


def _render_conditioned_search_domain(
    search: PiPLSSearchCV,
    *,
    output_path: Path,
) -> None:
    figure, axis, component_values, _, _ = _search_surface_figure(
        search,
        title="Pulp exhaustive search: conditional rank selection",
    )
    (
        _,
        exact_ranks,
        selected_ranks,
        _,
    ) = _relative_tolerance_path(
        search,
        relative_tolerance=DOMAIN_PREDICTOR_RELATIVE_TOLERANCE,
    )
    if np.array_equal(exact_ranks, selected_ranks):
        raise RuntimeError("Illustrative predictor-rank tolerance must change the path.")

    axis.plot(
        component_values,
        exact_ranks,
        linestyle="none",
        marker="o",
        markerfacecolor="black",
        markeredgecolor="white",
        markeredgewidth=1.8,
        markersize=10,
        label="Exact minimum at fixed h",
        zorder=3,
    )
    axis.plot(
        component_values,
        selected_ranks,
        marker="D",
        color="tab:orange",
        linewidth=1.5,
        markersize=6,
        label=(
            f"Smallest rank within "
            f"{100.0 * DOMAIN_PREDICTOR_RELATIVE_TOLERANCE:.0f}%"
        ),
        zorder=4,
    )
    axis.legend(loc="lower right", framealpha=0.0)
    _save_svg(figure, output_path)


def _render_conditioned_component_path(
    search: PiPLSSearchCV,
    *,
    output_path: Path,
) -> None:
    (
        component_values,
        _,
        selected_ranks,
        selected_mse,
    ) = _relative_tolerance_path(
        search,
        relative_tolerance=DOMAIN_PREDICTOR_RELATIVE_TOLERANCE,
    )
    exact_index = int(np.argmin(selected_mse))
    threshold = (
        1.0 + DOMAIN_COMPONENT_RELATIVE_TOLERANCE
    ) * float(selected_mse[exact_index])
    selected_index = int(np.flatnonzero(selected_mse <= threshold)[0])
    if selected_index == exact_index:
        raise RuntimeError("Illustrative component tolerance must change the selection.")

    figure, axis = _figure(figsize=(7.4, 4.8))
    axis.plot(component_values, selected_mse, "o-", label="Conditioned component path")
    axis.axhline(
        threshold,
        linewidth=1.0,
        linestyle="--",
        color="0.35",
        label=(
            f"{100.0 * DOMAIN_COMPONENT_RELATIVE_TOLERANCE:.0f}% "
            "relative threshold"
        ),
    )
    axis.scatter(
        [component_values[exact_index]],
        [selected_mse[exact_index]],
        marker="o",
        facecolors="none",
        edgecolors="black",
        linewidths=1.5,
        s=80,
        label="Exact path minimum",
        zorder=3,
    )
    axis.scatter(
        [component_values[selected_index]],
        [selected_mse[selected_index]],
        marker="D",
        color="tab:orange",
        s=70,
        label=f"Selected h = {component_values[selected_index]}",
        zorder=4,
    )
    for index, (h_value, rank_value, mse_value) in enumerate(
            zip(
                component_values,
                selected_ranks,
                selected_mse,
                strict=True,
            )
    ):
        if index == 0:
            xytext = (0, 7)
            va = "bottom"
        elif index == 1:
            xytext = (-4, -8)
            va = "top"
        else:
            xytext = (0, -7)
            va = "top"
            
        axis.annotate(
            rf"$r_\pi={int(rank_value)}$",
            (h_value, mse_value),
            xytext=xytext,
            textcoords="offset points",
            ha="center",
            va=va,
        )
    axis.set_title("Pulp conditioned component path")
    axis.set_xlabel(r"Paired-mode count $h$")
    axis.set_ylabel("Mean response-standardized CV-MSE")
    axis.set_xticks(component_values)
    axis.set_ylim(bottom=0.0, top=1.0)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    _save_svg(figure, output_path)


def _epv_rank(
    data: PiPLSDataset,
    search: PiPLSSearchCV,
) -> int:
    nominal = min(
        data.X.shape[1],
        int(np.ceil(data.X.shape[0] / EPV_SAMPLES_PER_PREDICTOR_RANK)),
    )
    return min(nominal, search.max_predictor_rank_)


def _render_epv_search_domain(
    search: PiPLSSearchCV,
    data: PiPLSDataset,
    *,
    output_path: Path,
) -> None:
    component_values, rank_values, surface = _candidate_surface(search)
    epv_rank = _epv_rank(data, search)
    if epv_rank not in rank_values:
        raise RuntimeError("Pulp EPV rank must lie in the exhaustive search domain.")

    figure, axis = _figure(figsize=(7.4, 5.0))
    admissible = np.isfinite(surface)
    background = np.where(admissible, 0.84, 0.94)
    axis.imshow(
        background,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=(
            0.5,
            float(component_values[-1]) + 0.5,
            0.5,
            float(rank_values[-1]) + 0.5,
        ),
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
    )

    epv_surface = np.full_like(surface, np.nan)
    epv_row = epv_rank - 1
    compatible = component_values <= epv_rank
    epv_surface[epv_row, compatible] = surface[epv_row, compatible]
    image = axis.imshow(
        np.ma.masked_invalid(epv_surface),
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=(
            0.5,
            float(component_values[-1]) + 0.5,
            0.5,
            float(rank_values[-1]) + 0.5,
        ),
        cmap="viridis",
        vmin=float(np.nanmin(surface)),
        vmax=float(np.nanmax(surface)),
    )
    axis.set_xticks(component_values)
    axis.set_yticks(rank_values)
    axis.set_xticks(
        np.arange(1.5, float(component_values[-1]) + 0.5),
        minor=True,
    )
    axis.set_yticks(
        np.arange(1.5, float(rank_values[-1]) + 0.5),
        minor=True,
    )
    axis.grid(which="minor", linewidth=0.5, color="white", alpha=0.65)
    axis.tick_params(which="minor", bottom=False, left=False)
    axis.set_xlabel(r"Paired-mode count $h$")
    axis.set_ylabel(r"Predictor rank $r_\pi$")
    axis.set_title(rf"Pulp EPV policy: fixed $r_\pi={epv_rank}$")
    axis.text(
        float(component_values[-1]) - 1.3,
        2.1,
        r"$r_\pi < h$" + "\nnot admissible",
        ha="center",
        va="center",
    )
    colorbar = figure.colorbar(image, ax=axis, shrink=0.86)
    colorbar.set_label("Mean response-standardized CV-MSE")
    _save_svg(figure, output_path)


def _render_component_path(
    component_path: PiPLSComponentPath,
    *,
    selected: PiPLSSelection | None,
    title: str,
    output_path: Path,
) -> None:
    figure, axis = _figure(figsize=(7.4, 4.8))
    axis.errorbar(
        component_path.n_components,
        component_path.cv_mse_mean,
        yerr=component_path.cv_mse_std,
        fmt="o-",
        capsize=4,
    )
    if selected is not None:
        axis.scatter(
            [selected.n_components],
            [selected.cv_mse_mean],
            marker="D",
            color="tab:orange",
            s=70,
            label=f"Chosen: {selected.n_components} components",
            zorder=3,
        )
        axis.legend()
    axis.set_title(title)
    axis.set_xlabel("Number of components")
    axis.set_ylabel("Mean response-standardized CV-MSE (±1 SD)")
    axis.set_xticks(component_path.n_components)
    upper = float(np.max(component_path.cv_mse_mean + component_path.cv_mse_std))
    axis.set_ylim(0.0, max(1.0, 1.05 * upper))
    axis.grid(axis="y", alpha=0.25)
    _save_svg(figure, output_path)


def _render_predictor_rank_profile(
    profile: PiPLSPredictorRankProfile,
    *,
    output_path: Path,
) -> None:
    figure, axis = _figure(figsize=(7.4, 4.8))
    axis.errorbar(
        profile.predictor_rank,
        profile.cv_mse_mean,
        yerr=profile.cv_mse_std,
        fmt="o-",
        capsize=4,
    )
    axis.scatter(
        [profile.selection.predictor_rank],
        [profile.selection.cv_mse_mean],
        marker="D",
        color="tab:orange",
        s=70,
        label=f"Selected rank: {profile.selection.predictor_rank}",
        zorder=3,
    )
    axis.set_title(
        "Pulp Π-PLS predictor-rank profile at "
        f"{profile.n_components} components"
    )
    axis.set_xlabel("Predictor rank")
    axis.set_ylabel("Mean response-standardized CV-MSE (±1 SD)")
    axis.set_xticks(profile.predictor_rank)
    upper = float(np.max(profile.cv_mse_mean + profile.cv_mse_std))
    axis.set_ylim(0.0, max(1.0, 1.05 * upper))
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    _save_svg(figure, output_path)


def _render_biplot(
    structure: LatentStructure,
    *,
    predictor_names: tuple[str, ...],
    output_path: Path,
) -> None:
    figure, axis = _figure(figsize=(9.0, 7.0))
    # --8<-- [start:render-pulp-biplot]
    biplot = biplot_coordinates(structure, components=(0, 1))
    plot_pulp_biplot(
        axis,
        biplot,
        predictor_names=predictor_names,
        title=r"Pulp $\Pi$-PLS score-loading biplot",
    )
    # --8<-- [end:render-pulp-biplot]
    _save_svg(figure, output_path)


def _render_predictor_directions(
    factors: PiPLSDisplayFactors,
    *,
    predictor_names: tuple[str, ...],
    n_components: int,
    output_path: Path,
) -> None:
    # --8<-- [start:render-pulp-predictor-directions]
    figure, axis = _figure(figsize=(10.0, 5.4))
    predictor_positions = np.arange(len(predictor_names))
    predictor_width = 0.8 / n_components
    for series, component in enumerate(range(n_components)):
        offset = (series - (n_components - 1) / 2.0) * predictor_width
        axis.bar(
            predictor_positions + offset,
            factors.predictor_directions[:, component],
            width=predictor_width,
            label=f"Component {component + 1}",
        )
    axis.axhline(0.0, linewidth=0.8, linestyle="--", color="0.45")
    axis.set_xticks(predictor_positions)
    axis.set_xticklabels(predictor_names)
    axis.set_xlabel("Predictor")
    axis.set_ylabel(r"Predictor direction $P_{:k}$")
    axis.legend()
    axis.tick_params(axis="x", labelrotation=45)
    for label in axis.get_xticklabels():
        label.set_horizontalalignment("right")
    # --8<-- [end:render-pulp-predictor-directions]
    _save_svg(figure, output_path)


def _render_weighted_response_directions(
    factors: PiPLSDisplayFactors,
    *,
    response_names: tuple[str, ...],
    n_components: int,
    output_path: Path,
) -> None:
    # --8<-- [start:render-pulp-weighted-response-directions]
    figure, axis = _figure(figsize=(8.2, 5.4))
    response_positions = np.arange(len(response_names))
    response_width = 0.8 / n_components
    for series, component in enumerate(range(n_components)):
        offset = (series - (n_components - 1) / 2.0) * response_width
        axis.bar(
            response_positions + offset,
            factors.weighted_response_directions[:, component],
            width=response_width,
            label=f"Component {component + 1}",
        )
    axis.axhline(0.0, linewidth=0.8, linestyle="--", color="0.45")
    axis.set_xticks(response_positions)
    axis.set_xticklabels(response_names)
    axis.set_xlabel("Response")
    axis.set_ylabel(r"Weighted response direction $D_kQ_{:k}$")
    axis.legend()
    axis.tick_params(axis="x", labelrotation=45)
    for label in axis.get_xticklabels():
        label.set_horizontalalignment("right")
    # --8<-- [end:render-pulp-weighted-response-directions]
    _save_svg(figure, output_path)


def _render_observed_vs_predicted(
    diagnostics: PredictionDiagnostics,
    *,
    response_names: tuple[str, ...],
    output_path: Path,
) -> None:
    # --8<-- [start:render-pulp-observed-vs-predicted]
    figure, axis = _figure(figsize=(6.4, 5.0))
    for response, response_name in enumerate(response_names):
        axis.scatter(
            diagnostics.observed_standardized[:, response],
            diagnostics.predicted_standardized[:, response],
            label=response_name,
            alpha=0.75,
        )
    values = np.concatenate(
        [
            diagnostics.observed_standardized.ravel(),
            diagnostics.predicted_standardized.ravel(),
        ]
    )
    lower = float(values.min())
    upper = float(values.max())
    margin = 0.05 * (upper - lower) if upper > lower else 1.0
    limits = (lower - margin, upper + margin)
    axis.plot(limits, limits, linewidth=1.0, linestyle="--", color="0.35")
    axis.set_xlim(limits)
    axis.set_ylim(limits)
    axis.set_xlabel("Observed response (standardized)")
    axis.set_ylabel("Predicted response (standardized)")
    axis.set_title(rf"Pulp $\Pi$-PLS — {diagnostics.prediction_kind}")
    axis.legend(title="Response", fontsize="small", ncols=2)
    # --8<-- [end:render-pulp-observed-vs-predicted]
    _save_svg(figure, output_path)


def _render_residuals_vs_predicted(
    diagnostics: PredictionDiagnostics,
    *,
    response_names: tuple[str, ...],
    output_path: Path,
) -> None:
    # --8<-- [start:render-pulp-residuals-vs-predicted]
    figure, axis = _figure(figsize=(6.4, 5.0))
    for response, response_name in enumerate(response_names):
        axis.scatter(
            diagnostics.predicted_standardized[:, response],
            diagnostics.residual_standardized[:, response],
            label=response_name,
            alpha=0.75,
        )
    predicted = diagnostics.predicted_standardized
    residual = diagnostics.residual_standardized
    predicted_span = float(predicted.max() - predicted.min())
    residual_span = float(residual.max() - residual.min())
    axis.set_xlim(
        float(predicted.min())
        - (0.05 * predicted_span if predicted_span > 0.0 else 1.0),
        float(predicted.max())
        + (0.05 * predicted_span if predicted_span > 0.0 else 1.0),
    )
    axis.set_ylim(
        float(residual.min())
        - (0.05 * residual_span if residual_span > 0.0 else 1.0),
        float(residual.max())
        + (0.05 * residual_span if residual_span > 0.0 else 1.0),
    )
    axis.axhline(0.0, linewidth=1.0, linestyle="--", color="0.35")
    axis.set_xlabel("Predicted response (standardized)")
    axis.set_ylabel("Standardized residual")
    axis.set_title(rf"Pulp $\Pi$-PLS — {diagnostics.prediction_kind}")
    axis.legend(title="Response", fontsize="small", ncols=2)
    # --8<-- [end:render-pulp-residuals-vs-predicted]
    _save_svg(figure, output_path)


def _render_oof_response_r2(
    diagnostics: PredictionDiagnostics,
    *,
    response_names: tuple[str, ...],
    output_path: Path,
) -> None:
    # --8<-- [start:render-pulp-oof-response-r2]
    figure, axis = _figure(figsize=(7.4, 5.0))
    positions = np.arange(len(response_names))
    axis.bar(positions, diagnostics.response_r2)
    axis.axhline(0.0, linewidth=0.8, linestyle="--", color="0.35")
    axis.set_xticks(positions)
    axis.set_xticklabels(response_names)
    axis.set_xlabel("Response")
    axis.set_ylabel(r"Response-wise OOF $R^2$")
    axis.set_ylim(*response_r2_ylim(diagnostics.response_r2))
    axis.set_title(rf"Pulp $\Pi$-PLS — {diagnostics.prediction_kind}")
    axis.tick_params(axis="x", labelrotation=45)
    for label in axis.get_xticklabels():
        label.set_horizontalalignment("right")
    # --8<-- [end:render-pulp-oof-response-r2]
    _save_svg(figure, output_path)


def _render_final_fit_observed_vs_predicted(
    diagnostics: PredictionDiagnostics,
    *,
    response_names: tuple[str, ...],
    output_path: Path,
) -> None:
    figure, axis = _figure(figsize=(6.2, 6.2))
    for response, response_name in enumerate(response_names):
        axis.scatter(
            diagnostics.observed_standardized[:, response],
            diagnostics.predicted_standardized[:, response],
            label=response_name,
            alpha=0.7,
        )

    limits_source = np.concatenate(
        [
            diagnostics.observed_standardized.ravel(),
            diagnostics.predicted_standardized.ravel(),
        ]
    )
    lower = float(limits_source.min())
    upper = float(limits_source.max())
    margin = 0.05 * (upper - lower) if upper > lower else 1.0
    limits = (lower - margin, upper + margin)
    axis.plot(limits, limits, linewidth=1.0, linestyle="--", color="0.35")
    axis.set_xlim(limits)
    axis.set_ylim(limits)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Observed response (standardized)")
    axis.set_ylabel("Fitted response (standardized)")
    axis.set_title(r"Final $\Pi$-PLS fit: observed versus fitted")
    axis.legend(title="Response", fontsize="small", ncols=2)
    axis.grid(alpha=0.2)
    _save_svg(figure, output_path)


def _render_final_fit_r2(
    diagnostics: PredictionDiagnostics,
    *,
    response_names: tuple[str, ...],
    output_path: Path,
) -> None:
    positions = np.arange(len(response_names))
    figure, axis = _figure(figsize=(8.2, 4.8))
    axis.bar(positions, diagnostics.response_r2)
    axis.axhline(0.0, linewidth=0.8, color="0.35")
    axis.set_xticks(positions)
    axis.set_xticklabels(response_names, rotation=45, ha="right")
    axis.set_xlabel("Response")
    axis.set_ylabel(r"Fitted $R^2$")
    axis.set_title(r"Final $\Pi$-PLS fit: response-wise $R^2$")
    axis.set_ylim(*response_r2_ylim(diagnostics.response_r2))
    axis.grid(axis="y", alpha=0.2)
    _save_svg(figure, output_path)


def _render_final_fit_residual_distribution(
    diagnostics: PredictionDiagnostics,
    *,
    output_path: Path,
) -> None:
    residuals = diagnostics.residual_standardized.ravel()
    residual_mean = float(np.mean(residuals))
    residual_std = float(np.std(residuals, ddof=1))

    figure, axis = _figure(figsize=(7.0, 4.8))
    axis.hist(
        residuals,
        bins=18,
        density=True,
        alpha=0.65,
        label="Standardized residuals",
    )
    if residual_std > 0.0:
        x_values = np.linspace(float(residuals.min()), float(residuals.max()), 300)
        normal_density = np.exp(
            -0.5 * ((x_values - residual_mean) / residual_std) ** 2
        ) / (residual_std * np.sqrt(2.0 * np.pi))
        axis.plot(
            x_values,
            normal_density,
            linewidth=1.5,
            label="Matched normal density",
        )
    axis.axvline(0.0, linewidth=0.8, linestyle="--", color="0.35")
    axis.set_xlabel("Standardized residual")
    axis.set_ylabel("Density")
    axis.set_title(r"Final $\Pi$-PLS fit: residual distribution")
    axis.legend()
    _save_svg(figure, output_path)

def _write_manifest(
    output_dir: Path,
    *,
    data: PiPLSDataset,
    search: PiPLSSearchCV,
    selection: PiPLSSelection,
    rank_profile: PiPLSPredictorRankProfile,
    report: PiPLSOOFReport,
    oof_diagnostics: PredictionDiagnostics,
    fitted_diagnostics: PredictionDiagnostics,
    displayed_components: tuple[int, ...],
) -> Path:
    figures = [
        {"filename": filename, "sha256": _sha256(output_dir / filename)}
        for filename in FIGURE_FILENAMES
    ]
    manifest = {
        "schema_version": 2,
        "dataset": {
            "id": data.metadata["dataset"]["id"],
            "version": data.metadata["dataset"]["version"],
            "source_doi": data.metadata["source"]["doi"],
            "license": data.metadata["license"]["identifier"],
            "resource_sha256": dict(data.metadata["integrity"]["resource_sha256"]),
            "array_sha256": dict(data.metadata["integrity"]["array_sha256"]),
        },
        "analysis": {
            "chosen_n_components": selection.n_components,
            "chosen_predictor_rank": selection.predictor_rank,
            "search_method": search.search_method,
            "search_is_exhaustive": search.search_is_exhaustive_,
            "max_predictor_rank": search.max_predictor_rank_,
            "evaluated_predictor_ranks": rank_profile.predictor_rank.tolist(),
            "predictor_rank_at_upper_boundary": bool(
                selection.predictor_rank == int(rank_profile.predictor_rank[-1])
            ),
            "displayed_components": [
                component + 1 for component in displayed_components
            ],
            "factor_sign_anchor": {"response": "TI", "sign": "positive"},
            "detailed_responses": list(data.target_names),
            "prediction_kind": oof_diagnostics.prediction_kind,
            "path_selection_explanation": {
                "predictor_rank_relative_tolerance": (
                    DOMAIN_PREDICTOR_RELATIVE_TOLERANCE
                ),
                "component_relative_tolerance": DOMAIN_COMPONENT_RELATIVE_TOLERANCE,
                "epv_samples_per_predictor_rank": EPV_SAMPLES_PER_PREDICTOR_RANK,
                "epv_predictor_rank": _epv_rank(data, search),
            },
            "cross_validation": {
                "splitter": type(CV).__name__,
                "n_splits": 5,
                "n_repeats": 10,
                "random_state": 0,
                "materialized_splits": search.n_splits_,
            },
            "oof_predictions_per_observation": int(
                report.oof_prediction_counts[0]
            ),
            "response_r2": [
                {"response": name, "value": float(value)}
                for name, value in zip(
                    data.target_names, oof_diagnostics.response_r2, strict=True
                )
            ],
        },
        "final_fit": {
            "prediction_kind": fitted_diagnostics.prediction_kind,
            "response_r2": [
                {"response": name, "value": float(value)}
                for name, value in zip(
                    data.target_names, fitted_diagnostics.response_r2, strict=True
                )
            ],
            "pooled_standardized_residuals": {
                "count": int(fitted_diagnostics.residual_standardized.size),
                "mean": float(np.mean(fitted_diagnostics.residual_standardized)),
                "sample_sd": float(
                    np.std(fitted_diagnostics.residual_standardized, ddof=1)
                ),
            },
        },
        "figures": figures,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def render_pulp_tutorial_assets(output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    """Generate the representative Pulp tutorial figures and return the manifest path."""

    output_dir = output_dir.resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    data = load_pulp()
    X, Y = data.X, data.Y
    predictor_names = data.feature_names
    response_names = data.target_names

    search = PiPLSSearchCV(cv=CV).fit(X, Y)
    if not search.search_is_exhaustive_:
        raise RuntimeError("The Pulp domain figure requires exhaustive search.")
    _render_search_domain(
        search,
        output_path=output_dir / "search_domain.svg",
    )
    _render_conditioned_search_domain(
        search,
        output_path=output_dir / "conditioned_search_domain.svg",
    )
    _render_conditioned_component_path(
        search,
        output_path=output_dir / "conditioned_component_path.svg",
    )
    _render_epv_search_domain(
        search,
        data,
        output_path=output_dir / "epv_search_domain.svg",
    )
    component_path = search.component_path_
    _render_component_path(
        component_path,
        selected=None,
        title=r"Pulp $\Pi$-PLS component path before selection",
        output_path=output_dir / "component_path.svg",
    )

    selection = search.select(n_components=3)
    rank_profile = search.predictor_rank_profile(selection.n_components)
    report = search.oof_report(X, Y, selection=selection)
    if not np.all(report.oof_prediction_counts == 10):
        raise RuntimeError(
            "Repeated Pulp CV must produce ten OOF predictions per observation."
        )
    oof_diagnostics = prediction_diagnostics(
        Y,
        report.oof_predictions,
        prediction_kind=PREDICTION_KIND,
    )
    _render_component_path(
        component_path,
        selected=selection,
        title=r"Pulp $\Pi$-PLS selected component path",
        output_path=output_dir / "selected_component_path.svg",
    )
    _render_predictor_rank_profile(
        rank_profile,
        output_path=output_dir / "predictor_rank_profile.svg",
    )

    model = search.refit(X, Y, selection=selection)
    fitted_diagnostics = prediction_diagnostics(
        Y,
        model.predict(X),
        prediction_kind="fitted values",
    )
    factors = pipls_display_factors(
        model.decomposition_,
        response_index=response_names.index("TI"),
        response_sign="positive",
    )
    structure = latent_structure(model)
    displayed_components = tuple(range(selection.n_components))

    _render_biplot(
        structure,
        predictor_names=predictor_names,
        output_path=output_dir / "biplot.svg",
    )
    _render_predictor_directions(
        factors,
        predictor_names=predictor_names,
        n_components=selection.n_components,
        output_path=output_dir / "predictor_directions.svg",
    )
    _render_weighted_response_directions(
        factors,
        response_names=response_names,
        n_components=selection.n_components,
        output_path=output_dir / "weighted_response_directions.svg",
    )
    _render_observed_vs_predicted(
        oof_diagnostics,
        response_names=response_names,
        output_path=output_dir / "observed_vs_predicted.svg",
    )
    _render_residuals_vs_predicted(
        oof_diagnostics,
        response_names=response_names,
        output_path=output_dir / "residuals_vs_predicted.svg",
    )
    _render_oof_response_r2(
        oof_diagnostics,
        response_names=response_names,
        output_path=output_dir / "oof_response_r2.svg",
    )
    _render_final_fit_observed_vs_predicted(
        fitted_diagnostics,
        response_names=response_names,
        output_path=output_dir / "final_fit_observed_vs_predicted.svg",
    )
    _render_final_fit_r2(
        fitted_diagnostics,
        response_names=response_names,
        output_path=output_dir / "final_fit_r2.svg",
    )
    _render_final_fit_residual_distribution(
        fitted_diagnostics,
        output_path=output_dir / "final_fit_residual_distribution.svg",
    )

    return _write_manifest(
        output_dir,
        data=data,
        search=search,
        selection=selection,
        rank_profile=rank_profile,
        report=report,
        oof_diagnostics=oof_diagnostics,
        fitted_diagnostics=fitted_diagnostics,
        displayed_components=displayed_components,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory receiving the generated SVG files and manifest.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = render_pulp_tutorial_assets(args.output_dir)
    print(f"Wrote Pulp tutorial figures to {manifest.parent}")


if __name__ == "__main__":
    main()
