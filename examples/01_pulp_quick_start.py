"""Fit Π-PLS on the package-owned Pulp data and plot fitted responses."""

from pathlib import Path

# --8<-- [start:load-pulp-data]
import matplotlib.pyplot as plt

from pipls import PiPLSSearchCV
from pipls.datasets import load_pulp
from pipls.inspection import prediction_diagnostics

data = load_pulp()
X, Y = data.X, data.Y
# --8<-- [end:load-pulp-data]

# --8<-- [start:fit-selected-pulp-model]
model = PiPLSSearchCV().fit(X, Y).refit(X, Y, rule="minimum_cv_mse")

diagnostics = prediction_diagnostics(
    Y,
    model.predict(X),
    prediction_kind="fitted values",
)
observed = diagnostics.observed_standardized.ravel()
fitted = diagnostics.predicted_standardized.ravel()
mean_standardized_rmse = float(diagnostics.standardized_rmse.mean())
# --8<-- [end:fit-selected-pulp-model]

# --8<-- [start:plot-standardized-fitted-values]
limits = [
    min(float(observed.min()), float(fitted.min())),
    max(float(observed.max()), float(fitted.max())),
]

figure, axis = plt.subplots(figsize=(5.8, 5.4), layout="constrained")
axis.scatter(observed, fitted)
axis.plot(limits, limits, "--", color="0.4")
axis.set_xlim(limits)
axis.set_ylim(limits)
axis.set_aspect("equal", adjustable="box")
axis.set_xlabel("Observed response, standardized")
axis.set_ylabel("Fitted response, standardized")
axis.set_title(
    rf"Pulp $\Pi$-PLS fit; mean standardized RMSE = {mean_standardized_rmse:.2f}"
)
axis.grid(alpha=0.2)
# --8<-- [end:plot-standardized-fitted-values]

output_path = Path(__file__).resolve().parent / "results" / "pulp_quick_start.pdf"
figure.savefig(output_path)
plt.close(figure)

print(
    "Selected model: "
    f"n_components={model.n_components}, predictor_rank={model.predictor_rank}"
)
print(f"Mean response-wise standardized RMSE: {mean_standardized_rmse:.3f}")
print(f"Wrote PDF figure to {output_path}")
