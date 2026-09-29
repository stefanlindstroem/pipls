"""Generate simplified Pulp and Tobacco PLS-family figures for the Home page."""

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
for directory in (SRC_DIR, EXAMPLES_DIR):
    path_entry = str(directory)
    if path_entry not in sys.path:
        sys.path.insert(0, path_entry)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "pipls-home-pls-comparison"
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from _support.pls_family_path_comparison import (  # noqa: E402
    PLSFamilyPathEvaluation,
    evaluate_pls_family_paths,
)
from matplotlib.figure import Figure  # noqa: E402

from pipls.datasets import PiPLSDataset, load_pulp, load_tobacco  # noqa: E402

DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "docs" / "assets" / "generated" / "home"
CASES = ("pulp", "tobacco")
FIGURE_FILENAMES = {
    "pulp": "pulp_component_parsimony.svg",
    "tobacco": "tobacco_component_parsimony.svg",
}
X_LABEL = "Nr of components"
Y_LABEL = "Mean response-standardized CV-MSE (±1 SD)"
PIPLS_LABEL = r"$\Pi$-PLS"
PLS_LABEL = "PLS"


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


def _load_case(case: str) -> PiPLSDataset:
    if case == "pulp":
        return load_pulp()
    if case == "tobacco":
        return load_tobacco()
    raise ValueError(f"Unknown Home comparison case: {case!r}.")


def _evaluate_case(case: str, data: PiPLSDataset) -> PLSFamilyPathEvaluation:
    return evaluate_pls_family_paths(
        case,
        data.X,
        data.Y,
        response_subspaces=("cross_covariance",),
    )


def _common_y_max(evaluations: dict[str, PLSFamilyPathEvaluation]) -> float:
    upper = 0.0
    for evaluation in evaluations.values():
        pipls_path = evaluation.pipls_searches["cross_covariance"].component_path_
        pls_path = evaluation.pls_path
        upper = max(
            upper,
            float(np.max(pipls_path.cv_mse_mean + pipls_path.cv_mse_std)),
            float(np.max(pls_path.cv_mse_mean + pls_path.cv_mse_std)),
        )
    return max(1.0, 1.05 * upper)


def _render_case(
    case: str,
    evaluation: PLSFamilyPathEvaluation,
    *,
    y_max: float,
    output_path: Path,
) -> None:
    pipls_path = evaluation.pipls_searches["cross_covariance"].component_path_
    pls_path = evaluation.pls_path

    figure, axis = plt.subplots(figsize=(4.0, 4.0), layout="constrained")
    axis.errorbar(
        pipls_path.n_components,
        pipls_path.cv_mse_mean,
        yerr=pipls_path.cv_mse_std,
        fmt="o-",
        capsize=4,
        label=PIPLS_LABEL,
    )
    axis.errorbar(
        pls_path.n_components,
        pls_path.cv_mse_mean,
        yerr=pls_path.cv_mse_std,
        fmt="s:",
        capsize=4,
        label=PLS_LABEL,
    )
    axis.set_title(case.capitalize())
    axis.set_xlabel(X_LABEL)
    axis.set_ylabel(Y_LABEL)
    axis.set_xticks(pipls_path.n_components)
    axis.set_ylim(0.0, y_max)
    axis.margins(x=0.05)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    _save_svg(figure, output_path)


def _case_manifest(
    case: str,
    data: PiPLSDataset,
    evaluation: PLSFamilyPathEvaluation,
    *,
    output_path: Path,
) -> dict[str, object]:
    search = evaluation.pipls_searches["cross_covariance"]
    pipls_path = search.component_path_
    pls_path = evaluation.pls_path
    estimator = search.estimator
    return {
        "dataset": {
            "id": data.metadata["dataset"]["id"],
            "version": data.metadata["dataset"]["version"],
        },
        "analysis": {
            "response_subspace": estimator.response_subspace,
            "search_method": search.search_method,
            "search_is_exhaustive": bool(search.search_is_exhaustive_),
            "max_predictor_rank": int(search.max_predictor_rank_),
            "svd_solver": estimator.svd_solver,
            "n_jobs": search.n_jobs,
            "n_splits": int(search.n_splits_),
            "n_components": pipls_path.n_components.tolist(),
            "predictor_rank": pipls_path.predictor_rank.tolist(),
            "pipls_cv_mse_mean": pipls_path.cv_mse_mean.tolist(),
            "pipls_cv_mse_std": pipls_path.cv_mse_std.tolist(),
            "pls_algorithm": pls_path.algorithm,
            "pls_cv_mse_mean": pls_path.cv_mse_mean.tolist(),
            "pls_cv_mse_std": pls_path.cv_mse_std.tolist(),
        },
        "figure": {
            "filename": output_path.name,
            "sha256": _sha256(output_path),
        },
    }


def render_home_pls_comparison_assets(output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    """Generate the simplified Home PLS figures and return the manifest path."""

    output_dir = output_dir.resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    data_by_case = {case: _load_case(case) for case in CASES}
    evaluations = {
        case: _evaluate_case(case, data_by_case[case])
        for case in CASES
    }
    y_max = _common_y_max(evaluations)

    for case in CASES:
        _render_case(
            case,
            evaluations[case],
            y_max=y_max,
            output_path=output_dir / FIGURE_FILENAMES[case],
        )

    case_records = {
        case: _case_manifest(
            case,
            data_by_case[case],
            evaluations[case],
            output_path=output_dir / FIGURE_FILENAMES[case],
        )
        for case in CASES
    }
    
    manifest = {
        "schema_version": 1,
        "comparison": {
            "response_subspace": "cross_covariance",
            "cv": {
                "n_splits": 5,
                "shuffle": True,
                "random_state": 0,
            },
            "x_label": X_LABEL,
            "y_label": Y_LABEL,
            "pipls_label": "Π-PLS",
            "pls_label": PLS_LABEL,
            "shared_y_max": y_max,
        },
        "cases": case_records,
        "figures": [
            case_records[case]["figure"]
            for case in CASES
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
        help="Directory receiving the generated SVG files and manifest.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = render_home_pls_comparison_assets(args.output_dir)
    print(f"Wrote Home PLS comparison figures to {manifest.parent}")


if __name__ == "__main__":
    main()
