"""Fit and inspect a Tobacco Π-PLS model directly in memory."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from _support.metric_plotting import response_r2_ylim
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from sklearn.model_selection import KFold

from pipls import PiPLSRegression, PiPLSSearchCV
from pipls.component_path import (
    PiPLSComponentPath,
    PiPLSPredictorRankProfile,
    PiPLSSelection,
)
from pipls.datasets import load_tobacco
from pipls.inspection import (
    LatentStructure,
    ObservationDiagnostics,
    PiPLSDisplayFactors,
    PredictionDiagnostics,
    latent_structure,
    observation_diagnostics,
    pipls_display_factors,
    prediction_diagnostics,
)

ANALYSIS_DIR = Path(__file__).resolve().parent / "results" / "tobacco_post_analysis"
DISPLAY_COMPONENT_COUNT = 4
RESPONSES_PER_PAGE = 5
PREDICTOR_RANK_RELATIVE_TOLERANCE = 0.20
COMPONENT_RELATIVE_TOLERANCE = 0.20
CV = KFold(n_splits=5, shuffle=True, random_state=0)


def _plot_component_path(
    path: PiPLSComponentPath,
    minimum: PiPLSSelection,
    selected: PiPLSSelection,
    *,
    cv_mse_threshold: float,
    component_relative_tolerance: float,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(
        figsize=(7.0, 4.5),
        layout="constrained",
    )
    axis.errorbar(
        path.n_components,
        path.cv_mse_mean,
        yerr=path.cv_mse_std,
        fmt="o-",
        capsize=4,
    )
    axis.scatter(
        [minimum.n_components],
        [minimum.cv_mse_mean],
        marker="X",
        color="0.35",
        s=70,
        label=(
            "Conditioned-path minimum: "
            f"{minimum.n_components} components"
        ),
        zorder=3,
    )
    axis.axhline(
        cv_mse_threshold,
        linewidth=1.2,
        linestyle="--",
        color="0.35",
        label=(
            f"{100.0 * component_relative_tolerance:.0f}% "
            "component-count threshold"
        ),
    )
    axis.scatter(
        [selected.n_components],
        [selected.cv_mse_mean],
        marker="D",
        color="tab:orange",
        s=70,
        label=(
            f"{100.0 * component_relative_tolerance:.0f}% "
            "component-count selection: "
            f"{selected.n_components} components, "
            f"predictor rank {selected.predictor_rank}"
        ),
        zorder=3,
    )
    axis.set_xlabel("Number of components")
    axis.set_ylabel("Mean response-standardized CV-MSE (±1 SD)")
    axis.set_title(r"Tobacco $\Pi$-PLS component path")
    axis.set_xticks(path.n_components)
    upper = max(
        float(np.max(path.cv_mse_mean + path.cv_mse_std)),
        float(cv_mse_threshold),
    )
    axis.set_ylim(0.0, max(1.0, 1.05 * upper))
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.savefig(output_path)
    plt.close(figure)


def _plot_predictor_rank_profile(
    profile: PiPLSPredictorRankProfile,
    *,
    cv_mse_threshold: float,
    predictor_rank_relative_tolerance: float,
    output_path: Path,
) -> None:
    reference = profile.reference_selection
    selected = profile.selection
    figure, axis = plt.subplots(
        figsize=(7.0, 4.5),
        layout="constrained",
    )
    axis.errorbar(
        profile.predictor_rank,
        profile.cv_mse_mean,
        yerr=profile.cv_mse_std,
        fmt="o-",
        capsize=4,
    )
    axis.scatter(
        [reference.predictor_rank],
        [reference.cv_mse_mean],
        marker="X",
        color="0.35",
        s=70,
        label=f"Exact conditional minimum: rank {reference.predictor_rank}",
        zorder=3,
    )
    axis.axhline(
        cv_mse_threshold,
        linewidth=1.2,
        linestyle="--",
        color="0.35",
        label=(
            f"{100.0 * predictor_rank_relative_tolerance:.0f}% "
            "predictor-rank threshold"
        ),
    )
    axis.scatter(
        [selected.predictor_rank],
        [selected.cv_mse_mean],
        marker="D",
        color="tab:orange",
        s=70,
        label=(
            f"{100.0 * predictor_rank_relative_tolerance:.0f}% "
            f"predictor-rank selection: rank {selected.predictor_rank}"
        ),
        zorder=3,
    )
    axis.set_xlabel("Predictor rank")
    axis.set_ylabel("Mean response-standardized CV-MSE (±1 SD)")
    axis.set_title(
        rf"Tobacco $\Pi$-PLS predictor-rank profile at "
        f"{profile.n_components} components"
    )
    axis.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    axis.set_ylim(bottom=0.0)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.savefig(output_path)
    plt.close(figure)


def _plot_pipls_factors(
    factors: PiPLSDisplayFactors,
    wavenumbers: NDArray[np.float64],
    response_names: list[str],
    *,
    selected_n_components: int,
    selected_predictor_rank: int,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12.0, 9.0),
        layout="constrained",
    )
    components = tuple(range(factors.n_components))
    component_labels = [f"Component {component + 1}" for component in components]
    for component, label in zip(components, component_labels, strict=True):
        axes[0, 0].plot(
            wavenumbers,
            factors.predictor_directions[:, component],
            label=label,
        )
    axes[0, 0].axhline(0.0, linewidth=0.8, linestyle="--", color="0.45")
    axes[0, 0].set_xlim(float(wavenumbers[0]), float(wavenumbers[-1]))
    axes[0, 0].set_xlabel("Wavenumber (cm$^{-1}$)")
    axes[0, 0].set_ylabel(r"Predictor direction $P_{:k}$")
    axes[0, 0].legend()

    dilation_positions = np.arange(factors.n_components)
    axes[0, 1].bar(dilation_positions, factors.dilation)
    axes[0, 1].set_xticks(dilation_positions)
    axes[0, 1].set_xticklabels(np.arange(1, factors.n_components + 1))
    axes[0, 1].set_xlabel("Component")
    axes[0, 1].set_ylabel(r"Dilation $D_k$")

    response_positions = np.arange(len(response_names))
    response_width = 0.8 / factors.n_components
    for component, label in zip(components, component_labels, strict=True):
        offset = (component - (factors.n_components - 1) / 2.0) * response_width
        axes[1, 0].bar(
            response_positions + offset,
            factors.response_directions[:, component],
            width=response_width,
            label=label,
        )
        axes[1, 1].bar(
            response_positions + offset,
            factors.weighted_response_directions[:, component],
            width=response_width,
            label=label,
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
        r"Tobacco $\Pi$-PLS factors "
        f"({selected_n_components} components, predictor rank {selected_predictor_rank})"
    )
    figure.savefig(output_path)
    plt.close(figure)


def _write_prediction_diagnostics_report(
    diagnostics: PredictionDiagnostics,
    response_names: list[str],
    response_pages: tuple[tuple[int, ...], ...],
    output_path: Path,
) -> None:
    with PdfPages(output_path) as report:
        for page_number, responses in enumerate(response_pages, start=1):
            figure, axes = plt.subplots(
                1,
                3,
                figsize=(13.0, 4.2),
                layout="constrained",
            )
            response_array = np.array(responses, dtype=np.int64)
            for response in responses:
                axes[0].scatter(
                    diagnostics.observed_standardized[:, response],
                    diagnostics.predicted_standardized[:, response],
                    label=response_names[response],
                    alpha=0.75,
                )
                axes[1].scatter(
                    diagnostics.predicted_standardized[:, response],
                    diagnostics.residual_standardized[:, response],
                    label=response_names[response],
                    alpha=0.75,
                )
            identity_values = np.concatenate(
                [
                    diagnostics.observed_standardized[:, response_array].ravel(),
                    diagnostics.predicted_standardized[:, response_array].ravel(),
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
            axes[1].axhline(0.0, linewidth=1.0, linestyle="--", color="0.35")
            axes[1].set_xlabel("Predicted response (standardized)")
            axes[1].set_ylabel("Standardized residual")
            positions = np.arange(len(responses))
            page_response_r2 = diagnostics.response_r2[response_array]
            axes[2].bar(positions, page_response_r2)
            axes[2].axhline(0.0, linewidth=0.8, linestyle="--", color="0.35")
            axes[2].set_xticks(positions)
            axes[2].set_xticklabels(
                [response_names[index] for index in responses],
                rotation=45,
                ha="right",
            )
            axes[2].set_xlabel("Response")
            axes[2].set_ylabel(r"Response-wise OOF $R^2$")
            axes[2].set_ylim(*response_r2_ylim(page_response_r2))
            if len(responses) > 1:
                axes[0].legend()
                axes[1].legend()
            figure.suptitle(
                rf"Tobacco $\Pi$-PLS prediction diagnostics — "
                f"response page {page_number}/{len(response_pages)} — "
                f"{diagnostics.prediction_kind}"
            )
            report.savefig(figure)
            plt.close(figure)


def _plot_latent_structure(
    structure: LatentStructure,
    observations: ObservationDiagnostics,
    wavenumbers: NDArray[np.float64],
    response_names: list[str],
    display_components: tuple[int, ...],
    *,
    selected_n_components: int,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12.0, 9.0),
        layout="constrained",
    )
    if selected_n_components >= 2:
        axes[0, 0].scatter(
            structure.x_scores[:, 0],
            structure.x_scores[:, 1],
            alpha=0.75,
        )
        axes[0, 0].axvline(0.0, linewidth=0.8, linestyle="--", color="0.45")
        axes[0, 0].set_xlabel("X score component 1")
        axes[0, 0].set_ylabel("X score component 2")
    else:
        observation_positions = np.arange(structure.x_scores.shape[0])
        axes[0, 0].scatter(
            observation_positions,
            structure.x_scores[:, 0],
            alpha=0.75,
        )
        axes[0, 0].set_xlabel("Observation")
        axes[0, 0].set_ylabel("X score component 1")
    axes[0, 0].axhline(0.0, linewidth=0.8, linestyle="--", color="0.45")

    for component in display_components:
        axes[0, 1].plot(
            wavenumbers,
            structure.x_loadings[:, component],
            label=f"Component {component + 1}",
        )
    axes[0, 1].axhline(0.0, linewidth=0.8, linestyle="--", color="0.45")
    axes[0, 1].set_xlabel("Wavenumber (cm$^{-1}$)")
    axes[0, 1].set_ylabel("X loading")
    axes[0, 1].legend()

    response_positions = np.arange(len(response_names))
    component_width = 0.8 / len(display_components)
    for series, component in enumerate(display_components):
        offset = (series - (len(display_components) - 1) / 2) * component_width
        axes[1, 0].bar(
            response_positions + offset,
            structure.y_loadings[:, component],
            width=component_width,
            label=f"Component {component + 1}",
        )
    axes[1, 0].axhline(0.0, linewidth=0.8, linestyle="--", color="0.45")
    axes[1, 0].set_xticks(response_positions)
    axes[1, 0].set_xticklabels(response_names, rotation=45, ha="right")
    axes[1, 0].set_xlabel("Response")
    axes[1, 0].set_ylabel("Y loading")
    axes[1, 0].legend()

    axes[1, 1].scatter(
        observations.score_distance,
        observations.x_reconstruction_residual,
        alpha=0.75,
    )
    axes[1, 1].set_xlabel("Score distance")
    axes[1, 1].set_ylabel("Squared X-reconstruction residual")
    figure.suptitle(
        r"Tobacco $\Pi$-PLS latent structure and observation diagnostics"
    )
    figure.savefig(output_path)
    plt.close(figure)


def _write_coefficients_report(
    structure: LatentStructure,
    wavenumbers: NDArray[np.float64],
    response_names: list[str],
    response_pages: tuple[tuple[int, ...], ...],
    output_path: Path,
) -> None:
    with PdfPages(output_path) as report:
        for page_number, responses in enumerate(response_pages, start=1):
            figure, axis = plt.subplots(
                figsize=(10.0, 5.0),
                layout="constrained",
            )
            for response in responses:
                axis.plot(
                    wavenumbers,
                    structure.coefficients[response],
                    label=response_names[response],
                )
            axis.axhline(0.0, linewidth=0.8, linestyle="--", color="0.45")
            axis.set_xlabel("Wavenumber (cm$^{-1}$)")
            axis.set_ylabel("Regression coefficient")
            axis.set_title(
                rf"Tobacco $\Pi$-PLS — "
                f"response page {page_number}/{len(response_pages)}"
            )
            axis.legend(title="Response")
            report.savefig(figure)
            plt.close(figure)


def main() -> None:
    data = load_tobacco()
    X = data.X
    Y = data.Y
    wavenumbers = np.asarray(data.feature_names, dtype=np.float64)
    response_names = list(data.target_names)
    response_pages = tuple(
        tuple(range(start, min(start + RESPONSES_PER_PAGE, len(response_names))))
        for start in range(0, len(response_names), RESPONSES_PER_PAGE)
    )

    # Optimize predictor rank, but use a 20% tolerance to prefer a smaller
    # spectral subspace when its CV performance remains close to the conditional
    # optimum. Component-count parsimony is applied separately below.
    search = PiPLSSearchCV(
        estimator=PiPLSRegression(
            n_components=1,
            predictor_rank=1,
            svd_solver="full",
        ),
        predictor_rank_relative_tolerance=PREDICTOR_RANK_RELATIVE_TOLERANCE,
        search_method="adaptive",
        # Bound parallelism to four workers for this expensive high-dimensional search.
        # n_jobs changes execution only; the candidate domain and CV splits are unchanged.
        n_jobs=4,
        cv=CV,
    ).fit(X, Y)
    path = search.component_path_
    selection = search.select(
        rule="minimum_cv_mse",
        relative_tolerance=COMPONENT_RELATIVE_TOLERANCE,
    )
    rank_profile = search.predictor_rank_profile(selection.n_components)

    # Inspect the same tolerance-derived selection before final full-data fitting.
    report = search.oof_report(X, Y, selection=selection)
    minimum = selection.reference_minimum
    component_cv_mse_threshold = selection.cv_mse_threshold
    component_relative_tolerance = selection.relative_tolerance
    predictor_rank_evidence = rank_profile.predictor_rank_evidence
    if (
        minimum is None
        or component_cv_mse_threshold is None
        or component_relative_tolerance is None
        or predictor_rank_evidence is None
    ):
        raise RuntimeError("The tolerance selections lack their reference evidence.")
    predictor_rank_cv_mse_threshold = -predictor_rank_evidence.score_threshold
    predictor_rank_reference = rank_profile.reference_selection
    predictor_rank_selection = rank_profile.selection
    display_components = tuple(
        range(min(DISPLAY_COMPONENT_COUNT, selection.n_components))
    )
    oof_predictions = report.oof_predictions
    diagnostics = prediction_diagnostics(
        Y,
        oof_predictions,
        prediction_kind="selection-conditioned OOF predictions",
    )

    # Refit the accepted selection on all development observations.
    model = search.refit(X, Y, selection=selection)

    # Calculate immutable fitted-model inspection results.
    factors = pipls_display_factors(model.decomposition_)
    structure = latent_structure(model)
    observations = observation_diagnostics(model, X)

    # Render the final reports from completed public result objects.
    _plot_component_path(
        path,
        minimum,
        selection,
        cv_mse_threshold=component_cv_mse_threshold,
        component_relative_tolerance=component_relative_tolerance,
        output_path=ANALYSIS_DIR / "component_path.pdf",
    )
    _plot_predictor_rank_profile(
        rank_profile,
        cv_mse_threshold=predictor_rank_cv_mse_threshold,
        predictor_rank_relative_tolerance=(
            predictor_rank_evidence.relative_tolerance
        ),
        output_path=ANALYSIS_DIR / "predictor_rank_profile.pdf",
    )
    _plot_pipls_factors(
        factors,
        wavenumbers,
        response_names,
        selected_n_components=selection.n_components,
        selected_predictor_rank=selection.predictor_rank,
        output_path=ANALYSIS_DIR / "pipls_factors.pdf",
    )
    _write_prediction_diagnostics_report(
        diagnostics,
        response_names,
        response_pages,
        ANALYSIS_DIR / "prediction_diagnostics.pdf",
    )
    _plot_latent_structure(
        structure,
        observations,
        wavenumbers,
        response_names,
        display_components,
        selected_n_components=selection.n_components,
        output_path=ANALYSIS_DIR / "latent_structure.pdf",
    )
    _write_coefficients_report(
        structure,
        wavenumbers,
        response_names,
        response_pages,
        ANALYSIS_DIR / "coefficients.pdf",
    )

    print(f"X shape: {X.shape}; Y shape: {Y.shape}")
    print(
        "Predictor-rank choice at "
        f"n_components={selection.n_components}: "
        f"exact optimum r_pi={predictor_rank_reference.predictor_rank}, "
        f"{100.0 * predictor_rank_evidence.relative_tolerance:.0f}% "
        f"retained r_pi={predictor_rank_selection.predictor_rank}, "
        f"CV-MSE threshold={predictor_rank_cv_mse_threshold:.6g}"
    )
    print(
        "Component-count choice on the conditioned path: "
        f"exact minimum h={minimum.n_components}, "
        f"{100.0 * component_relative_tolerance:.0f}% "
        f"retained h={selection.n_components}, "
        f"CV-MSE threshold={component_cv_mse_threshold:.6g}"
    )
    print(
        "Final Π-PLS model: "
        f"n_components={model.n_components}, predictor_rank={selection.predictor_rank}"
    )
    print(f"Wrote PDF figures to {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
