"""Generate the deterministic fitted-value figure for the Pulp quick start."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPOSITORY_ROOT / "src"
path_entry = str(SRC_DIR)
if path_entry not in sys.path:
    sys.path.insert(0, path_entry)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "pipls-quick-start-tutorial"
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from pipls import PiPLSSearchCV  # noqa: E402
from pipls.datasets import load_pulp  # noqa: E402
from pipls.inspection import prediction_diagnostics  # noqa: E402

DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "docs" / "assets" / "generated" / "quick_start"
FIGURE_FILENAMES = ("observed_vs_fitted.svg",)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_svg(figure: Figure, path: Path) -> None:
    figure.savefig(
        path,
        format="svg",
        metadata={"Creator": "Pi-PLS repository", "Date": None},
    )
    plt.close(figure)


def render_quick_start_tutorial_assets(output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    """Generate the quick-start figure and return the semantic manifest path."""

    output_dir = output_dir.resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    data = load_pulp()
    X, Y = data.X, data.Y
    search = PiPLSSearchCV().fit(X, Y)
    model = search.refit(X, Y, rule="minimum_cv_mse")
    diagnostics = prediction_diagnostics(
        Y,
        model.predict(X),
        prediction_kind="fitted values",
    )
    observed = diagnostics.observed_standardized.ravel()
    fitted = diagnostics.predicted_standardized.ravel()
    mean_standardized_rmse = float(diagnostics.standardized_rmse.mean())
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
    figure_path = output_dir / FIGURE_FILENAMES[0]
    _save_svg(figure, figure_path)

    manifest = {
        "schema_version": 1,
        "dataset": {
            "id": data.metadata["dataset"]["id"],
            "version": data.metadata["dataset"]["version"],
        },
        "analysis": {
            "selected_n_components": model.n_components,
            "selected_predictor_rank": model.predictor_rank,
            "search_method": search.search_method,
            "search_is_exhaustive": search.search_is_exhaustive_,
            "max_predictor_rank": search.max_predictor_rank_,
            "prediction_kind": diagnostics.prediction_kind,
            "mean_standardized_rmse": mean_standardized_rmse,
        },
        "figures": [
            {
                "filename": figure_path.name,
                "sha256": _sha256(figure_path),
            }
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory receiving the generated SVG and manifest.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = render_quick_start_tutorial_assets(args.output_dir)
    print(f"Wrote quick-start tutorial figure to {manifest.parent}")


if __name__ == "__main__":
    main()
