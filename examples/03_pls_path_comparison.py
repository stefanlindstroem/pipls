"""Compare Π-PLS response policies and ordinary PLS on four matched-CV cases."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from _support.pls_component_path import PLSComponentPath
from _support.pls_family_path_comparison import (
    COMPARISON_CASES,
    SYNTHETIC_STRESS_CASE,
    CVSplit,
    evaluate_pls_family_paths,
)

from pipls import PiPLSSearchCV
from pipls.datasets import (
    load_pulp,
    load_sugarcane,
    load_tobacco,
    make_synthetic_data,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "pls_path_comparison"


@dataclass(frozen=True)
class SyntheticStressSpec:
    """Fixed geometry for the near-saturated synthetic comparison case."""

    n_samples: int = 25
    n_features: int = 40
    n_targets: int = 10
    n_shared: int = 5
    n_predictor_specific: int = 15
    n_response_specific: int = 0
    noise: float = 0.2
    random_state: int = 0


SYNTHETIC_STRESS_SPEC = SyntheticStressSpec()


def _make_synthetic_stress_case() -> tuple[np.ndarray, np.ndarray]:
    """Generate the fixed near-saturated synthetic stress case."""

    spec = SYNTHETIC_STRESS_SPEC
    return make_synthetic_data(
        n_samples=spec.n_samples,
        n_features=spec.n_features,
        n_targets=spec.n_targets,
        n_shared=spec.n_shared,
        n_predictor_specific=spec.n_predictor_specific,
        n_response_specific=spec.n_response_specific,
        noise=spec.noise,
        random_state=spec.random_state,
    )


@dataclass(frozen=True)
class DatasetComparison:
    """In-memory result of one matched PLS-family path comparison."""

    dataset: str
    cv_splits: list[CVSplit]
    cross_covariance_search: PiPLSSearchCV
    least_squares_search: PiPLSSearchCV
    pls_path: PLSComponentPath
    output_path: Path


def _load_dataset(dataset: str) -> tuple[np.ndarray, np.ndarray]:
    if dataset == "pulp":
        return load_pulp(return_X_y=True)
    if dataset == "sugarcane":
        return load_sugarcane(return_X_y=True)
    if dataset == "tobacco":
        return load_tobacco(return_X_y=True)
    if dataset == SYNTHETIC_STRESS_CASE:
        return _make_synthetic_stress_case()
    raise ValueError(f"Unknown comparison case: {dataset!r}.")


def _case_title(dataset: str) -> str:
    if dataset == SYNTHETIC_STRESS_CASE:
        return "Synthetic stress-case PLS-family component-path comparison"
    return f"{dataset.capitalize()} PLS-family component-path comparison"


def _print_case_context(dataset: str, *, X: np.ndarray, Y: np.ndarray) -> None:
    if dataset != SYNTHETIC_STRESS_CASE:
        print(f"{dataset.capitalize()}: X shape={X.shape}, Y shape={Y.shape}")
        return

    spec = SYNTHETIC_STRESS_SPEC
    fold_training_size = spec.n_samples - (spec.n_samples // 5)
    centered_rank_limit = fold_training_size - 1
    print(f"Synthetic stress case: X shape={X.shape}, Y shape={Y.shape}")
    print(
        "  latent dimensions: "
        f"shared={spec.n_shared}, predictor-specific={spec.n_predictor_specific}, "
        f"response-specific={spec.n_response_specific}"
    )
    print(f"  noise SD: X={spec.noise}, Y={spec.noise}")
    print(
        f"  fold training size: {fold_training_size}; centered predictor rank cannot exceed "
        f"{centered_rank_limit}"
    )


def run_dataset(dataset: str) -> DatasetComparison:
    """Run one three-way matched-CV comparison and write its PDF figure."""

    X, Y = _load_dataset(dataset)
    evaluation = evaluate_pls_family_paths(
        dataset,
        X,
        Y,
        response_subspaces=("cross_covariance", "least_squares"),
    )
    cv_splits = evaluation.cv_splits
    cross_covariance_search = evaluation.pipls_searches["cross_covariance"]
    least_squares_search = evaluation.pipls_searches["least_squares"]
    cross_covariance_path = cross_covariance_search.component_path_
    least_squares_path = least_squares_search.component_path_
    pls_path = evaluation.pls_path

    figure, axis = plt.subplots(figsize=(8.4, 5.2), layout="constrained")
    axis.errorbar(
        cross_covariance_path.n_components,
        cross_covariance_path.cv_mse_mean,
        yerr=cross_covariance_path.cv_mse_std,
        fmt="o-",
        capsize=4,
        label=r"$\Pi$-PLS (cross-covariance)",
    )
    axis.errorbar(
        least_squares_path.n_components,
        least_squares_path.cv_mse_mean,
        yerr=least_squares_path.cv_mse_std,
        fmt="^--",
        capsize=4,
        label=r"$\Pi$-PLS (least squares)",
    )
    axis.errorbar(
        pls_path.n_components,
        pls_path.cv_mse_mean,
        yerr=pls_path.cv_mse_std,
        fmt="s:",
        capsize=4,
        label=f"PLS ({pls_path.algorithm})",
    )
    upper = max(
        float(np.max(cross_covariance_path.cv_mse_mean + cross_covariance_path.cv_mse_std)),
        float(np.max(least_squares_path.cv_mse_mean + least_squares_path.cv_mse_std)),
        float(np.max(pls_path.cv_mse_mean + pls_path.cv_mse_std)),
    )
    axis.set_title(_case_title(dataset))
    axis.set_xlabel("Nr of components")
    axis.set_ylabel("Mean response-standardized CV-MSE (±1 SD)")
    axis.set_xticks(cross_covariance_path.n_components)
    axis.set_ylim(0.0, max(1.0, 1.05 * upper))
    axis.margins(x=0.05)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()

    output_path = RESULTS_DIR / f"{dataset}_component_path_comparison.pdf"
    figure.savefig(output_path)
    plt.close(figure)

    _print_case_context(dataset, X=X, Y=Y)
    print(f"  shared validation protocol: {len(cv_splits)} materialized folds")
    print(
        "  cross_covariance (peer-reviewed default) predictor ranks: "
        f"{cross_covariance_path.predictor_rank.tolist()}"
    )
    print(
        "  least_squares (software extension; not part of the peer-reviewed "
        "publication) predictor ranks: "
        f"{least_squares_path.predictor_rank.tolist()}"
    )
    print(
        "  path CV-MSE values are model-development evidence, not independent "
        "post-selection validation."
    )

    return DatasetComparison(
        dataset=dataset,
        cv_splits=cv_splits,
        cross_covariance_search=cross_covariance_search,
        least_squares_search=least_squares_search,
        pls_path=pls_path,
        output_path=output_path,
    )


def main(argv: Sequence[str] | None = None) -> dict[str, DatasetComparison]:
    """Run one requested comparison case or all three datasets plus the synthetic case."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=COMPARISON_CASES,
        help="Run only one comparison case; the default runs all four.",
    )
    args = parser.parse_args(argv)
    datasets = COMPARISON_CASES if args.dataset is None else (args.dataset,)
    results = {dataset: run_dataset(dataset) for dataset in datasets}
    print(f"Wrote results to {RESULTS_DIR}")
    return results


if __name__ == "__main__":
    main()
