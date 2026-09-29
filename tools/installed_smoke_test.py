"""Smoke-test a clean installation of Π-PLS outside the repository checkout."""

from __future__ import annotations

import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np

import pipls
from pipls import PiPLSRegression, PiPLSSearchCV
from pipls.datasets import load_pulp, load_sugarcane, load_tobacco


def main() -> None:
    repository = Path(sys.argv[1]).resolve()
    artifact_label = sys.argv[2]
    package_file = Path(pipls.__file__).resolve()
    source_root = (repository / "src").resolve()
    environment_root = Path(sys.prefix).resolve()

    if package_file.is_relative_to(source_root):
        raise AssertionError(
            f"pipls imported from the repository checkout: {package_file}"
        )
    if not package_file.is_relative_to(environment_root):
        raise AssertionError(
            f"pipls did not import from the clean environment: {package_file}"
        )
    if version("pipls") != pipls.__version__:
        raise AssertionError("Installed metadata and package versions disagree.")

    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 4))
    Y = np.column_stack((X[:, 0] + X[:, 1], X[:, 2] - X[:, 3]))

    model = PiPLSRegression(n_components=1, predictor_rank=2).fit(X, Y)
    if model.response_subspace != "cross_covariance":
        raise AssertionError(
            "Installed fixed estimator does not use cross-covariance response "
            "selection by default."
        )
    prediction = model.predict(X[:2])
    if prediction.shape != (2, 2) or not np.isfinite(prediction).all():
        raise AssertionError("Installed fixed-estimator prediction failed.")

    least_squares_model = PiPLSRegression(
        n_components=1,
        predictor_rank=2,
        response_subspace="least_squares",
    ).fit(X, Y)
    least_squares_prediction = least_squares_model.predict(X[:2])
    if (
        least_squares_model.response_subspace != "least_squares"
        or least_squares_prediction.shape != (2, 2)
        or not np.isfinite(least_squares_prediction).all()
    ):
        raise AssertionError(
            "Installed least-squares response-subspace estimator failed."
        )

    search_defaults = PiPLSSearchCV()
    if search_defaults.search_method != "exhaustive":
        raise AssertionError("Installed search does not use exhaustive coverage by default.")
    if search_defaults.predictor_rank_values is not None:
        raise AssertionError("Installed search does not optimize predictor rank by default.")
    if search_defaults.max_predictor_rank is not None:
        raise AssertionError("Installed search unexpectedly imposes a default rank cap.")
    if search_defaults.samples_per_predictor_rank != 10.0:
        raise AssertionError("Installed EPV default does not use c=10.")

    search = PiPLSSearchCV(
        n_components_values=(1,),
        cv=2,
        n_jobs=1,
    ).fit(X, Y)
    if search.max_predictor_rank_ != 4 or not search.search_is_exhaustive_:
        raise AssertionError("Installed search did not resolve the full hard-feasible rank domain.")
    evaluated_ranks = set(search.cv_results_["predictor_rank"].tolist())
    if evaluated_ranks != {1, 2, 3, 4}:
        raise AssertionError(
            f"Installed exhaustive search evaluated unexpected ranks: {sorted(evaluated_ranks)}"
        )
    selection = search.select(n_components=1)
    refitted = search.refit(X, Y, selection=selection)
    if not np.isfinite(refitted.predict(X[:2])).all():
        raise AssertionError("Installed search and refit workflow failed.")

    epv_search = PiPLSSearchCV(
        n_components_values=(1,),
        predictor_rank_values="epv",
        cv=2,
        n_jobs=1,
    ).fit(X, Y)
    epv_selection = epv_search.select(n_components=1)
    if epv_selection.predictor_rank != 2 or epv_selection.predictor_rank_policy != "epv":
        raise AssertionError("Installed EPV policy did not resolve the expected fixed rank.")

    least_squares_search = PiPLSSearchCV(
        estimator=PiPLSRegression(
            n_components=1,
            predictor_rank=2,
            response_subspace="least_squares",
        ),
        n_components_values=(1,),
        predictor_rank_values=(2,),
        cv=2,
        n_jobs=1,
    ).fit(X, Y)
    least_squares_selection = least_squares_search.select(n_components=1)
    least_squares_refit = least_squares_search.refit(
        X,
        Y,
        selection=least_squares_selection,
    )
    if least_squares_refit.response_subspace != "least_squares":
        raise AssertionError(
            "Installed search did not preserve least-squares response selection."
        )
    if not np.isfinite(least_squares_refit.predict(X[:2])).all():
        raise AssertionError(
            "Installed least-squares search and refit workflow failed."
        )

    expected_shapes = {
        "pulp": ((46, 14), (46, 8)),
        "sugarcane": ((57, 1721), (57, 4)),
        "tobacco": ((347, 1557), (347, 13)),
    }
    for name, loader in (
        ("pulp", load_pulp),
        ("sugarcane", load_sugarcane),
        ("tobacco", load_tobacco),
    ):
        dataset = loader()
        if (dataset.X.shape, dataset.Y.shape) != expected_shapes[name]:
            raise AssertionError(f"Installed {name} dataset has unexpected shapes.")

    print(
        f"{artifact_label} installation passed: "
        f"pipls {pipls.__version__} from {package_file}"
    )


if __name__ == "__main__":
    main()
