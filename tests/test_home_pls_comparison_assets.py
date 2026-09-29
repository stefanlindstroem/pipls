from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

FIGURE_FILENAMES = {
    "pulp": "pulp_component_parsimony.svg",
    "tobacco": "tobacco_component_parsimony.svg",
}


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _load_renderer(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    repository = _repository_root()
    monkeypatch.syspath_prepend(str(repository / "examples"))
    path = repository / "tools" / "render_home_pls_comparison.py"
    spec = importlib.util.spec_from_file_location("home_pls_comparison_renderer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load renderer module from {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fake_evaluation(case: str) -> SimpleNamespace:
    components = np.arange(1, 4, dtype=np.intp)
    if case == "pulp":
        pipls_mean = np.array([0.52, 0.31, 0.23])
        pls_mean = np.array([0.71, 0.49, 0.39])
        ranks = np.array([9, 9, 9], dtype=np.intp)
        search_method = "exhaustive"
        search_is_exhaustive = True
        max_rank = 14
        svd_solver = "auto"
        n_jobs = None
    else:
        pipls_mean = np.array([0.68, 0.49, 0.34])
        pls_mean = np.array([0.76, 0.67, 0.58])
        ranks = np.array([35, 35, 35], dtype=np.intp)
        search_method = "adaptive"
        search_is_exhaustive = False
        max_rank = 346
        svd_solver = "full"
        n_jobs = 4

    pipls_path = SimpleNamespace(
        n_components=components,
        predictor_rank=ranks,
        cv_mse_mean=pipls_mean,
        cv_mse_std=np.array([0.08, 0.06, 0.05]),
    )
    pls_path = SimpleNamespace(
        n_components=components,
        cv_mse_mean=pls_mean,
        cv_mse_std=np.array([0.07, 0.06, 0.05]),
        algorithm="NIPALS",
        n_splits=5,
    )
    estimator = SimpleNamespace(
        response_subspace="cross_covariance",
        svd_solver=svd_solver,
    )
    search = SimpleNamespace(
        estimator=estimator,
        search_method=search_method,
        search_is_exhaustive_=search_is_exhaustive,
        max_predictor_rank_=max_rank,
        n_jobs=n_jobs,
        n_splits_=5,
        component_path_=pipls_path,
    )
    return SimpleNamespace(
        case=case,
        cv_splits=[object() for _ in range(5)],
        pipls_searches={"cross_covariance": search},
        pls_path=pls_path,
    )


def test_home_renderer_writes_simplified_matched_protocol_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_renderer(monkeypatch)
    calls: list[tuple[str, tuple[str, ...]]] = []

    def _fake_evaluate(
        case: str,
        X: object,
        Y: object,
        *,
        response_subspaces: tuple[str, ...],
    ) -> SimpleNamespace:
        del X, Y
        calls.append((case, response_subspaces))
        return _fake_evaluation(case)

    monkeypatch.setattr(module, "evaluate_pls_family_paths", _fake_evaluate)
    output_dir = tmp_path / "home"
    manifest_path = module.render_home_pls_comparison_assets(output_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert calls == [
        ("pulp", ("cross_covariance",)),
        ("tobacco", ("cross_covariance",)),
    ]
    assert manifest["schema_version"] == 1
    comparison = manifest["comparison"]
    assert comparison["response_subspace"] == "cross_covariance"
    assert comparison["cv"] == {"n_splits": 5, "shuffle": True, "random_state": 0}
    assert math.isfinite(float(comparison["shared_y_max"]))

    assert set(manifest["cases"]) == {"pulp", "tobacco"}
    assert manifest["cases"]["pulp"]["dataset"] == {"id": "pulp", "version": "1"}
    assert manifest["cases"]["tobacco"]["dataset"] == {
        "id": "tobacco",
        "version": "1",
    }

    pulp_analysis = manifest["cases"]["pulp"]["analysis"]
    assert pulp_analysis["response_subspace"] == "cross_covariance"
    assert pulp_analysis["search_method"] == "exhaustive"
    assert pulp_analysis["search_is_exhaustive"] is True
    assert pulp_analysis["svd_solver"] == "auto"
    assert pulp_analysis["n_jobs"] is None

    tobacco_analysis = manifest["cases"]["tobacco"]["analysis"]
    assert tobacco_analysis["response_subspace"] == "cross_covariance"
    assert tobacco_analysis["search_method"] == "adaptive"
    assert tobacco_analysis["search_is_exhaustive"] is False
    assert tobacco_analysis["svd_solver"] == "full"
    assert tobacco_analysis["n_jobs"] == 4

    for case in ("pulp", "tobacco"):
        analysis = manifest["cases"][case]["analysis"]
        assert analysis["n_splits"] == 5
        assert analysis["pls_algorithm"] == "NIPALS"
        assert analysis["n_components"] == [1, 2, 3]
        assert len(analysis["predictor_rank"]) == 3
        for field in (
            "pipls_cv_mse_mean",
            "pipls_cv_mse_std",
            "pls_cv_mse_mean",
            "pls_cv_mse_std",
        ):
            assert len(analysis[field]) == 3
            assert all(math.isfinite(float(value)) for value in analysis[field])

        figure = manifest["cases"][case]["figure"]
        assert figure["filename"] == FIGURE_FILENAMES[case]
        figure_path = output_dir / figure["filename"]
        ET.parse(figure_path)
        assert figure["sha256"] == _sha256(figure_path)

    assert set(path.name for path in output_dir.iterdir()) == {
        *FIGURE_FILENAMES.values(),
        "manifest.json",
    }
