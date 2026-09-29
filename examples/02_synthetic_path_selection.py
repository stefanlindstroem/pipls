"""Inspect a component path, select one pair, and fit synthetic Π-PLS."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# --8<-- [start:import-synthetic-kfold]
from sklearn.model_selection import KFold

# --8<-- [end:import-synthetic-kfold]
from pipls import PiPLSSearchCV
from pipls.component_path import PiPLSComponentPath, PiPLSSelection
from pipls.datasets import make_synthetic_data
from pipls.inspection import prediction_diagnostics

ANALYSIS_DIR = Path(__file__).resolve().parent / "results" / "synthetic_tutorial"
# --8<-- [start:define-synthetic-cv]
CV = KFold(n_splits=5, shuffle=True, random_state=0)
# --8<-- [end:define-synthetic-cv]

# --8<-- [start:define-synthetic-component-path-plotter]
def _plot_component_path(
    path: PiPLSComponentPath,
    *,
    selection: PiPLSSelection | None,
    title: str,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7.0, 4.5), layout="constrained")
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


# --8<-- [end:define-synthetic-component-path-plotter]


# --8<-- [start:generate-synthetic-data]
X, Y = make_synthetic_data(
    n_samples=180,
    n_features=8,
    n_targets=3,
    n_shared=2,
    n_predictor_specific=2,
    n_response_specific=1,
    noise=(0.2, 0.25),
    random_state=0,
)
X_train, X_test = X[:120], X[120:]
Y_train, Y_test = Y[:120], Y[120:]
# --8<-- [end:generate-synthetic-data]

# --8<-- [start:fit-synthetic-search]
search = PiPLSSearchCV(cv=CV).fit(X_train, Y_train)
# --8<-- [end:fit-synthetic-search]

# --8<-- [start:inspect-synthetic-component-path]
path = search.component_path_
# --8<-- [end:inspect-synthetic-component-path]

# --8<-- [start:choose-synthetic-selection]
CHOSEN_N_COMPONENTS = 2
selection = search.select(n_components=CHOSEN_N_COMPONENTS)
# --8<-- [end:choose-synthetic-selection]

# --8<-- [start:inspect-synthetic-selected-evidence]
rank_profile = search.predictor_rank_profile(selection.n_components)
# --8<-- [end:inspect-synthetic-selected-evidence]

# --8<-- [start:refit-synthetic-model]
model = search.refit(
    X_train,
    Y_train,
    selection=selection,
)
# --8<-- [end:refit-synthetic-model]

# --8<-- [start:evaluate-synthetic-predictions]
test_predictions = model.predict(X_test)
diagnostics = prediction_diagnostics(
    Y_test,
    test_predictions,
    prediction_kind="external test predictions",
)
# --8<-- [end:evaluate-synthetic-predictions]

# --8<-- [start:plot-synthetic-component-path]
_plot_component_path(
    path,
    selection=None,
    title=r"Synthetic $\Pi$-PLS component path before selection",
    output_path=ANALYSIS_DIR / "component_path.pdf",
)
# --8<-- [end:plot-synthetic-component-path]

# --8<-- [start:plot-synthetic-selected-component-path]
_plot_component_path(
    path,
    selection=selection,
    title=r"Synthetic $\Pi$-PLS selected component path",
    output_path=ANALYSIS_DIR / "selected_component_path.pdf",
)
# --8<-- [end:plot-synthetic-selected-component-path]

# --8<-- [start:plot-synthetic-rank-profile]
figure, axis = plt.subplots(figsize=(7.0, 4.5), layout="constrained")
axis.errorbar(
    rank_profile.predictor_rank,
    rank_profile.cv_mse_mean,
    yerr=rank_profile.cv_mse_std,
    fmt="o-",
    capsize=4,
)
axis.scatter(
    [rank_profile.selection.predictor_rank],
    [rank_profile.selection.cv_mse_mean],
    marker="D",
    color="tab:orange",
    s=70,
    label=f"Selected rank: {rank_profile.selection.predictor_rank}",
    zorder=3,
)
axis.set_xlabel("Predictor rank")
axis.set_ylabel("Mean response-standardized CV-MSE (±1 SD)")
axis.set_title(
    rf"Synthetic $\Pi$-PLS predictor-rank profile at "
    f"{selection.n_components} components"
)
axis.set_xticks(rank_profile.predictor_rank)
upper = float(np.max(rank_profile.cv_mse_mean + rank_profile.cv_mse_std))
axis.set_ylim(0.0, max(1.0, 1.05 * upper))
axis.grid(axis="y", alpha=0.25)
axis.legend()
figure.savefig(ANALYSIS_DIR / "predictor_rank_profile.pdf")
plt.close(figure)
# --8<-- [end:plot-synthetic-rank-profile]

# --8<-- [start:plot-synthetic-predictions]
figure, axis = plt.subplots(figsize=(6.2, 5.0), layout="constrained")
for response in range(Y.shape[1]):
    name = f"y_{response:03d}"
    axis.scatter(
        diagnostics.observed_standardized[:, response],
        diagnostics.predicted_standardized[:, response],
        label=name,
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
axis.set_title(rf"Synthetic $\Pi$-PLS — {diagnostics.prediction_kind}")
axis.legend(title="Response")
figure.savefig(ANALYSIS_DIR / "observed_vs_predicted.pdf")
plt.close(figure)
# --8<-- [end:plot-synthetic-predictions]

print("Synthetic Π-PLS path-selection example")
print(f"Training data: X{X_train.shape}, Y{Y_train.shape}")
print(f"Independent test data: X{X_test.shape}, Y{Y_test.shape}")
print("Known latent structure: 2 shared and 2 predictor-specific directions")
print(
    "Selected fixed model: "
    f"n_components={selection.n_components}, "
    f"predictor_rank={selection.predictor_rank}"
)
print(f"External-test R^2: {model.score(X_test, Y_test):.3f}")
print(f"Wrote PDF figures to {ANALYSIS_DIR}")
