"""Fit and inspect a Pulp Pi-PLS model directly in memory."""

# --8<-- [start:pulp-tutorial-setup]
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from _support.metric_plotting import response_r2_ylim
from _support.pulp_biplot import plot_pulp_biplot
from sklearn.model_selection import RepeatedKFold

from pipls import PiPLSSearchCV
from pipls.component_path import (
    PiPLSComponentPath,
    PiPLSPredictorRankProfile,
    PiPLSSelection,
)
from pipls.datasets import load_pulp
from pipls.inspection import (
    LatentStructure,
    PiPLSDisplayFactors,
    PredictionDiagnostics,
    biplot_coordinates,
    latent_structure,
    pipls_display_factors,
    prediction_diagnostics,
)

ANALYSIS_DIR = Path(__file__).resolve().parent / "results" / "pulp_post_analysis"
CV = RepeatedKFold(n_splits=5, n_repeats=10, random_state=0)
# --8<-- [end:pulp-tutorial-setup]


# --8<-- [start:define-pulp-component-path-plotter]
def _plot_component_path(
    path: PiPLSComponentPath,
    *,
    selection: PiPLSSelection | None,
    title: str,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(
        figsize=(7.4, 4.8),
        layout="constrained",
    )
    axis.errorbar(
        path.n_components,
        path.cv_mse_mean,
        yerr=path.cv_mse_std,
        fmt="o-",
        capsize=4,
    )
    if selection is not None:
        axis.scatter(
            [selection.n_components],
            [selection.cv_mse_mean],
            marker="D",
            color="tab:orange",
            s=70,
            label=f"Chosen: {selection.n_components} components",
            zorder=3,
        )
        axis.legend()
    axis.set_xlabel("Number of components")
    axis.set_ylabel("Mean response-standardized CV-MSE (±1 SD)")
    axis.set_title(title)
    axis.set_xticks(path.n_components)
    upper = float(np.max(path.cv_mse_mean + path.cv_mse_std))
    axis.set_ylim(0.0, max(1.0, 1.05 * upper))
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(output_path)
    plt.close(figure)


# --8<-- [end:define-pulp-component-path-plotter]


def _plot_predictor_rank_profile(
    profile: PiPLSPredictorRankProfile,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7.4, 4.8), layout="constrained")
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
    axis.set_xlabel("Predictor rank")
    axis.set_ylabel("Mean response-standardized CV-MSE (±1 SD)")
    axis.set_title(
        "Pulp Π-PLS predictor-rank profile at "
        f"{profile.n_components} components"
    )
    axis.set_xticks(profile.predictor_rank)
    upper = float(np.max(profile.cv_mse_mean + profile.cv_mse_std))
    axis.set_ylim(0.0, max(1.0, 1.05 * upper))
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.savefig(output_path)
    plt.close(figure)


def _plot_pipls_factors(
    factors: PiPLSDisplayFactors,
    *,
    predictor_names: tuple[str, ...],
    response_names: tuple[str, ...],
    selection: PiPLSSelection,
    output_path: Path,
) -> None:
    display_components = tuple(range(selection.n_components))
    component_indices = np.array(display_components, dtype=np.int64)
    component_labels = [
        f"Component {component + 1}" for component in display_components
    ]
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(13.0, 9.0),
        layout="constrained",
    )

    predictor_positions = np.arange(len(predictor_names))
    predictor_width = 0.8 / selection.n_components
    for series, component in enumerate(display_components):
        offset = (
            series - (selection.n_components - 1) / 2.0
        ) * predictor_width
        axes[0, 0].bar(
            predictor_positions + offset,
            factors.predictor_directions[:, component],
            width=predictor_width,
            label=component_labels[series],
        )
    axes[0, 0].axhline(0.0, linewidth=0.8, linestyle="--", color="0.45")
    axes[0, 0].set_xticks(predictor_positions)
    axes[0, 0].set_xticklabels(predictor_names, rotation=45, ha="right")
    axes[0, 0].set_xlabel("Predictor")
    axes[0, 0].set_ylabel(r"Predictor direction $P_{:k}$")
    axes[0, 0].legend()

    dilation_positions = np.arange(selection.n_components)
    axes[0, 1].bar(dilation_positions, factors.dilation[component_indices])
    axes[0, 1].set_xticks(dilation_positions)
    axes[0, 1].set_xticklabels(np.arange(1, selection.n_components + 1))
    axes[0, 1].set_xlabel("Component")
    axes[0, 1].set_ylabel(r"Dilation $D_k$")

    response_positions = np.arange(len(response_names))
    response_width = 0.8 / selection.n_components
    for series, component in enumerate(display_components):
        offset = (
            series - (selection.n_components - 1) / 2.0
        ) * response_width
        axes[1, 0].bar(
            response_positions + offset,
            factors.response_directions[:, component],
            width=response_width,
            label=component_labels[series],
        )
        axes[1, 1].bar(
            response_positions + offset,
            factors.weighted_response_directions[:, component],
            width=response_width,
            label=component_labels[series],
        )
    for axis, ylabel in (
        (axes[1, 0], r"Response direction $Q_{:k}$"),
        (axes[1, 1], r"Weighted response direction $D_kQ_{:k}$"),
    ):
        axis.axhline(0.0, linewidth=0.8, linestyle="--", color="0.45")
        axis.set_xticks(response_positions)
        axis.set_xticklabels(response_names, rotation=45, ha="right")
        axis.set_xlabel("Response")
        axis.set_ylabel(ylabel)
        axis.legend()

    figure.suptitle(
        r"Pulp $\Pi$-PLS factors with TI-positive orientation "
        f"({selection.n_components} components, "
        f"predictor rank {selection.predictor_rank})"
    )
    figure.savefig(output_path)
    plt.close(figure)


def _plot_latent_structure(
    structure: LatentStructure,
    *,
    predictor_names: tuple[str, ...],
    response_names: tuple[str, ...],
    n_components: int,
    output_path: Path,
) -> None:
    display_components = tuple(range(n_components))
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(13.0, 10.0),
        layout="constrained",
    )

    score_components = (0, 1)
    axes[0, 0].scatter(
        structure.x_scores[:, score_components[0]],
        structure.x_scores[:, score_components[1]],
        alpha=0.75,
    )
    axes[0, 0].axhline(0.0, linewidth=0.8, linestyle="--", color="0.45")
    axes[0, 0].axvline(0.0, linewidth=0.8, linestyle="--", color="0.45")
    axes[0, 0].set_xlabel(f"X score component {score_components[0] + 1}")
    axes[0, 0].set_ylabel(f"X score component {score_components[1] + 1}")

    # --8<-- [start:plot-pulp-biplot]
    biplot = biplot_coordinates(structure, components=(0, 1))
    plot_pulp_biplot(
        axes[0, 1],
        biplot,
        predictor_names=predictor_names,
    )
    # --8<-- [end:plot-pulp-biplot]

    predictor_positions = np.arange(len(predictor_names))
    component_width = 0.8 / n_components
    for series, component in enumerate(display_components):
        offset = (series - (n_components - 1) / 2) * component_width
        axes[1, 0].bar(
            predictor_positions + offset,
            structure.x_loadings[:, component],
            width=component_width,
            label=f"Component {component + 1}",
        )
    axes[1, 0].axhline(0.0, linewidth=0.8, linestyle="--", color="0.45")
    axes[1, 0].set_xticks(predictor_positions)
    axes[1, 0].set_xticklabels(predictor_names)
    axes[1, 0].set_xlabel("Predictor")
    axes[1, 0].set_ylabel("X loading")
    axes[1, 0].legend()

    response_positions = np.arange(len(response_names))
    for series, component in enumerate(display_components):
        offset = (series - (n_components - 1) / 2) * component_width
        axes[1, 1].bar(
            response_positions + offset,
            structure.y_loadings[:, component],
            width=component_width,
            label=f"Component {component + 1}",
        )
    axes[1, 1].axhline(0.0, linewidth=0.8, linestyle="--", color="0.45")
    axes[1, 1].set_xticks(response_positions)
    axes[1, 1].set_xticklabels(response_names)
    axes[1, 1].set_xlabel("Response")
    axes[1, 1].set_ylabel("Y loading")
    axes[1, 1].legend()
    for axis in (axes[1, 0], axes[1, 1]):
        axis.tick_params(axis="x", labelrotation=45)
        for label in axis.get_xticklabels():
            label.set_horizontalalignment("right")

    figure.suptitle(r"Pulp $\Pi$-PLS latent structure")
    figure.savefig(output_path)
    plt.close(figure)


def _plot_prediction_diagnostics(
    diagnostics: PredictionDiagnostics,
    *,
    response_names: tuple[str, ...],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(14.0, 4.6),
        layout="constrained",
    )
    for response, response_name in enumerate(response_names):
        axes[0].scatter(
            diagnostics.observed_standardized[:, response],
            diagnostics.predicted_standardized[:, response],
            label=response_name,
            alpha=0.75,
        )
        axes[1].scatter(
            diagnostics.predicted_standardized[:, response],
            diagnostics.residual_standardized[:, response],
            label=response_name,
            alpha=0.75,
        )

    identity_values = np.concatenate(
        [
            diagnostics.observed_standardized.ravel(),
            diagnostics.predicted_standardized.ravel(),
        ]
    )
    identity_lower = float(identity_values.min())
    identity_upper = float(identity_values.max())
    identity_margin = (
        0.05 * (identity_upper - identity_lower)
        if identity_upper > identity_lower
        else 1.0
    )
    identity_limits = (
        identity_lower - identity_margin,
        identity_upper + identity_margin,
    )
    axes[0].plot(
        identity_limits,
        identity_limits,
        linewidth=1.0,
        linestyle="--",
        color="0.35",
    )
    axes[0].set_xlim(identity_limits)
    axes[0].set_ylim(identity_limits)
    axes[0].set_xlabel("Observed response (standardized)")
    axes[0].set_ylabel("Predicted response (standardized)")

    predicted = diagnostics.predicted_standardized
    residual = diagnostics.residual_standardized
    predicted_span = float(predicted.max() - predicted.min())
    residual_span = float(residual.max() - residual.min())
    axes[1].set_xlim(
        float(predicted.min())
        - (0.05 * predicted_span if predicted_span > 0.0 else 1.0),
        float(predicted.max())
        + (0.05 * predicted_span if predicted_span > 0.0 else 1.0),
    )
    axes[1].set_ylim(
        float(residual.min())
        - (0.05 * residual_span if residual_span > 0.0 else 1.0),
        float(residual.max())
        + (0.05 * residual_span if residual_span > 0.0 else 1.0),
    )
    axes[1].axhline(0.0, linewidth=1.0, linestyle="--", color="0.35")
    axes[1].set_xlabel("Predicted response (standardized)")
    axes[1].set_ylabel("Standardized residual")

    positions = np.arange(len(response_names))
    axes[2].bar(positions, diagnostics.response_r2)
    axes[2].axhline(0.0, linewidth=0.8, linestyle="--", color="0.35")
    axes[2].set_xticks(positions)
    axes[2].set_xticklabels(response_names)
    axes[2].set_xlabel("Response")
    axes[2].set_ylabel(r"Response-wise OOF $R^2$")
    axes[2].set_ylim(*response_r2_ylim(diagnostics.response_r2))
    axes[0].legend(title="Response")
    axes[1].legend(title="Response")
    axes[2].tick_params(axis="x", labelrotation=45)
    for label in axes[2].get_xticklabels():
        label.set_horizontalalignment("right")

    figure.suptitle(
        rf"Pulp $\Pi$-PLS prediction diagnostics — {diagnostics.prediction_kind}"
    )
    figure.savefig(output_path)
    plt.close(figure)


def _plot_final_fit_observed_vs_predicted(
    diagnostics: PredictionDiagnostics,
    *,
    response_names: tuple[str, ...],
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(6.2, 6.2), layout="constrained")
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
    figure.savefig(output_path)
    plt.close(figure)


def _plot_final_fit_r2(
    diagnostics: PredictionDiagnostics,
    *,
    response_names: tuple[str, ...],
    output_path: Path,
) -> None:
    positions = np.arange(len(response_names))
    figure, axis = plt.subplots(figsize=(8.2, 4.8), layout="constrained")
    axis.bar(positions, diagnostics.response_r2)
    axis.axhline(0.0, linewidth=0.8, color="0.35")
    axis.set_xticks(positions)
    axis.set_xticklabels(response_names, rotation=45, ha="right")
    axis.set_xlabel("Response")
    axis.set_ylabel(r"Fitted $R^2$")
    axis.set_title(r"Final $\Pi$-PLS fit: response-wise $R^2$")
    axis.set_ylim(*response_r2_ylim(diagnostics.response_r2))
    axis.grid(axis="y", alpha=0.2)
    figure.savefig(output_path)
    plt.close(figure)


def _plot_final_fit_residual_distribution(
    diagnostics: PredictionDiagnostics,
    *,
    output_path: Path,
) -> None:
    residuals = diagnostics.residual_standardized.ravel()
    residual_mean = float(np.mean(residuals))
    residual_std = float(np.std(residuals, ddof=1))

    figure, axis = plt.subplots(figsize=(7.0, 4.8), layout="constrained")
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
    figure.savefig(output_path)
    plt.close(figure)


def _plot_coefficients(
    structure: LatentStructure,
    *,
    predictor_names: tuple[str, ...],
    response_names: tuple[str, ...],
    output_path: Path,
) -> None:
    response_indices = tuple(range(len(response_names)))
    predictor_positions = np.arange(len(predictor_names))
    response_width = 0.8 / len(response_indices)
    figure, axis = plt.subplots(figsize=(10.0, 5.4), layout="constrained")
    for series, response in enumerate(response_indices):
        offset = (series - (len(response_indices) - 1) / 2) * response_width
        axis.bar(
            predictor_positions + offset,
            structure.coefficients[response],
            width=response_width,
            label=response_names[response],
        )
    axis.axhline(0.0, linewidth=0.8, linestyle="--", color="0.45")
    axis.set_xticks(predictor_positions)
    axis.set_xticklabels(predictor_names)
    axis.set_xlabel("Predictor")
    axis.set_ylabel("Regression coefficient")
    axis.legend(title="Response")
    axis.tick_params(axis="x", labelrotation=45)
    for label in axis.get_xticklabels():
        label.set_horizontalalignment("right")
    figure.savefig(output_path)
    plt.close(figure)


# --8<-- [start:load-pulp-data]
data = load_pulp()
X, Y = data.X, data.Y
predictor_names = data.feature_names
response_names = data.target_names
# --8<-- [end:load-pulp-data]

# --8<-- [start:inspect-pulp-component-path]
search = PiPLSSearchCV(cv=CV).fit(X, Y)
path = search.component_path_
# --8<-- [end:inspect-pulp-component-path]

# --8<-- [start:choose-pulp-selection]
CHOSEN_N_COMPONENTS = 3
selection = search.select(n_components=CHOSEN_N_COMPONENTS)
# --8<-- [end:choose-pulp-selection]

# --8<-- [start:inspect-pulp-selected-evidence]
rank_profile = search.predictor_rank_profile(selection.n_components)
# --8<-- [end:inspect-pulp-selected-evidence]

# --8<-- [start:pulp-oof-predictions]
report = search.oof_report(
    X,
    Y,
    selection=selection,
)
oof_predictions = report.oof_predictions
# --8<-- [end:pulp-oof-predictions]

# --8<-- [start:pulp-oof-inspection-results]
oof_diagnostics = prediction_diagnostics(
    Y,
    oof_predictions,
    prediction_kind="selection-conditioned OOF predictions",
)
# --8<-- [end:pulp-oof-inspection-results]

# --8<-- [start:fit-pulp-model]
model = search.refit(
    X,
    Y,
    selection=selection,
)
# --8<-- [end:fit-pulp-model]

# --8<-- [start:pulp-final-fit-diagnostics]
fitted_predictions = model.predict(X)
fitted_diagnostics = prediction_diagnostics(
    Y,
    fitted_predictions,
    prediction_kind="fitted values",
)
# --8<-- [end:pulp-final-fit-diagnostics]

# --8<-- [start:pulp-fitted-model-inspection-results]
factors = pipls_display_factors(
    model.decomposition_,
    response_index=response_names.index("TI"),
    response_sign="positive",
)
structure = latent_structure(model)
# --8<-- [end:pulp-fitted-model-inspection-results]

# --8<-- [start:plot-pulp-component-path]
_plot_component_path(
    path,
    selection=None,
    title=r"Pulp $\Pi$-PLS component path before selection",
    output_path=ANALYSIS_DIR / "component_path.pdf",
)
# --8<-- [end:plot-pulp-component-path]

# --8<-- [start:plot-pulp-selected-component-path]
_plot_component_path(
    path,
    selection=selection,
    title=r"Pulp $\Pi$-PLS selected component path",
    output_path=ANALYSIS_DIR / "selected_component_path.pdf",
)
# --8<-- [end:plot-pulp-selected-component-path]

# --8<-- [start:plot-pulp-rank-profile]
_plot_predictor_rank_profile(
    rank_profile,
    ANALYSIS_DIR / "predictor_rank_profile.pdf",
)
# --8<-- [end:plot-pulp-rank-profile]

_plot_pipls_factors(
    factors,
    predictor_names=predictor_names,
    response_names=response_names,
    selection=selection,
    output_path=ANALYSIS_DIR / "pipls_factors.pdf",
)
_plot_latent_structure(
    structure,
    predictor_names=predictor_names,
    response_names=response_names,
    n_components=selection.n_components,
    output_path=ANALYSIS_DIR / "latent_structure.pdf",
)

# --8<-- [start:plot-pulp-prediction-diagnostics]
_plot_prediction_diagnostics(
    oof_diagnostics,
    response_names=response_names,
    output_path=ANALYSIS_DIR / "prediction_diagnostics.pdf",
)
# --8<-- [end:plot-pulp-prediction-diagnostics]

_plot_coefficients(
    structure,
    predictor_names=predictor_names,
    response_names=response_names,
    output_path=ANALYSIS_DIR / "coefficients.pdf",
)

# --8<-- [start:plot-pulp-final-fit-diagnostics]
_plot_final_fit_observed_vs_predicted(
    fitted_diagnostics,
    response_names=response_names,
    output_path=ANALYSIS_DIR / "final_fit_observed_vs_predicted.pdf",
)
_plot_final_fit_r2(
    fitted_diagnostics,
    response_names=response_names,
    output_path=ANALYSIS_DIR / "final_fit_r2.pdf",
)
_plot_final_fit_residual_distribution(
    fitted_diagnostics,
    output_path=ANALYSIS_DIR / "final_fit_residual_distribution.pdf",
)
# --8<-- [end:plot-pulp-final-fit-diagnostics]

print(f"X shape: {X.shape}; Y shape: {Y.shape}")
print(
    "Selected Pi-PLS: "
    f"n_components={model.n_components}, predictor_rank={selection.predictor_rank}"
)
print(
    "Predictor-rank profile: "
    f"evaluated {rank_profile.predictor_rank[0]} to "
    f"{rank_profile.predictor_rank[-1]}; "
    f"selected rank {selection.predictor_rank}"
)
print(f"Wrote PDF figures to {ANALYSIS_DIR}")
