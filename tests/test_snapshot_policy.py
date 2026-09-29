from __future__ import annotations

import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _commit(root: Path, message: str) -> None:
    subprocess.run(
        ["git", "commit", "--quiet", "-m", message],
        cwd=root,
        check=True,
    )


def _create_snapshot_test_repository(path: Path) -> Path:
    root = path / "repository"
    (root / "tools").mkdir(parents=True)
    shutil.copy2(
        _repository_root() / "tools" / "create_snapshot.py",
        root / "tools" / "create_snapshot.py",
    )
    (root / "README.md").write_text("# Snapshot fixture\n", encoding="utf-8")
    (root / ".gitignore").write_text(
        "\n".join(
            (
                "__pycache__/",
                "*.py[cod]",
                ".pytest_cache/",
                ".mypy_cache/",
                ".ruff_cache/",
                ".coverage",
                "coverage.xml",
                "htmlcov/",
                "build/",
                "dist/",
                "*.egg-info/",
                "docs/_build/",
                "docs/assets/generated/",
                "site/",
                "examples/results/*",
                "!examples/results/.gitkeep",
                "!examples/results/*/",
                "!examples/results/*/.gitkeep",
                "*-snapshot.tar.gz",
                "",
            )
        ),
        encoding="utf-8",
    )

    result_directories = (
        "examples/results",
        "examples/results/pls_path_comparison",
        "examples/results/synthetic_tutorial",
        "examples/results/pulp_post_analysis",
        "examples/results/sugarcane_post_analysis",
        "examples/results/tobacco_post_analysis",
    )
    for relative in result_directories:
        directory = root / relative
        directory.mkdir(parents=True, exist_ok=True)
        (directory / ".gitkeep").touch()

    subprocess.run(["git", "init", "--quiet"], cwd=root, check=True)
    subprocess.run(
        ["git", "config", "user.email", "snapshot@example.invalid"],
        cwd=root,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Snapshot Test"],
        cwd=root,
        check=True,
    )
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    _commit(root, "Create snapshot fixture")
    return root


def _run_snapshot(root: Path, archive: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(root / "tools" / "create_snapshot.py"), str(archive)],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )


def test_snapshot_has_committed_repository_contents_at_archive_root(tmp_path: Path) -> None:
    root = _create_snapshot_test_repository(tmp_path)
    archive = root / "fixture-snapshot.tar.gz"

    completed = _run_snapshot(root, archive)
    assert completed.returncode == 0, completed.stderr

    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        names = {member.name.removeprefix("./") for member in members}
        metadata_member = handle.extractfile("SNAPSHOT_INFO")
        assert metadata_member is not None
        metadata = metadata_member.read().decode("utf-8")

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert "README.md" in names
    assert "SNAPSHOT_INFO" in names
    assert not any(name.startswith(f"{root.name}/") for name in names)
    assert f"commit: {commit}" in metadata
    assert "dirty: false" in metadata

    expected_placeholders = {
        "examples/results/.gitkeep",
        "examples/results/pls_path_comparison/.gitkeep",
        "examples/results/synthetic_tutorial/.gitkeep",
        "examples/results/pulp_post_analysis/.gitkeep",
        "examples/results/sugarcane_post_analysis/.gitkeep",
        "examples/results/tobacco_post_analysis/.gitkeep",
    }
    archived_results = {
        member.name.removeprefix("./")
        for member in members
        if member.isfile()
        and member.name.removeprefix("./").startswith("examples/results/")
    }
    assert archived_results == expected_placeholders


@pytest.mark.parametrize("state", ["modified", "staged", "untracked"])
def test_snapshot_refuses_dirty_worktrees(tmp_path: Path, state: str) -> None:
    root = _create_snapshot_test_repository(tmp_path)
    if state == "untracked":
        (root / "notes.txt").write_text("untracked\n", encoding="utf-8")
    else:
        (root / "README.md").write_text(f"# {state}\n", encoding="utf-8")
        if state == "staged":
            subprocess.run(["git", "add", "README.md"], cwd=root, check=True)

    archive = tmp_path / f"{state}.tar.gz"
    completed = _run_snapshot(root, archive)

    assert completed.returncode != 0
    assert "Refusing to create a snapshot from a dirty worktree" in completed.stderr
    assert not archive.exists()


@pytest.mark.parametrize(
    "relative_path",
    [
        "__pycache__/module.cpython-310.pyc",
        ".pytest_cache/state",
        ".mypy_cache/state.json",
        ".ruff_cache/state",
        ".coverage",
        "coverage.xml",
        "htmlcov/index.html",
        "build/lib/pipls.py",
        "dist/pipls-0.1.0.whl",
        "src/pipls.egg-info/PKG-INFO",
        "docs/_build/index.html",
        "docs/assets/generated/tutorial.svg",
        "site/index.html",
        "examples/results/generated.pdf",
    ],
)
def test_snapshot_refuses_tracked_cache_and_generated_artifacts(
    tmp_path: Path,
    relative_path: str,
) -> None:
    root = _create_snapshot_test_repository(tmp_path)
    artifact = root / relative_path
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text("generated fixture\n", encoding="utf-8")
    subprocess.run(["git", "add", "-f", relative_path], cwd=root, check=True)
    _commit(root, "Commit prohibited artifact")

    archive = tmp_path / "snapshot.tar.gz"
    completed = _run_snapshot(root, archive)

    assert completed.returncode != 0
    assert "tracked cache or generated artifacts" in completed.stderr
    assert relative_path in completed.stderr
    assert not archive.exists()


def test_snapshot_ignores_untracked_ignored_artifacts(tmp_path: Path) -> None:
    root = _create_snapshot_test_repository(tmp_path)
    ignored_files = (
        ".pytest_cache/state",
        "__pycache__/module.cpython-310.pyc",
        "docs/assets/generated/tutorial.svg",
        "site/index.html",
        "build/lib/pipls.py",
        "examples/results/generated.pdf",
    )
    for relative_path in ignored_files:
        artifact = root / relative_path
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("ignored fixture\n", encoding="utf-8")

    archive = tmp_path / "snapshot.tar.gz"
    completed = _run_snapshot(root, archive)
    assert completed.returncode == 0, completed.stderr

    with tarfile.open(archive, "r:gz") as handle:
        names = {member.name.removeprefix("./") for member in handle.getmembers()}
    assert set(ignored_files).isdisjoint(names)


def test_example_result_tree_contains_placeholders_and_ignored_outputs() -> None:
    repository_root = _repository_root()
    result_root = repository_root / "examples" / "results"
    expected_placeholders = {
        ".gitkeep",
        "pls_path_comparison/.gitkeep",
        "pulp_post_analysis/.gitkeep",
        "sugarcane_post_analysis/.gitkeep",
        "synthetic_tutorial/.gitkeep",
        "tobacco_post_analysis/.gitkeep",
    }
    result_files = {
        path.relative_to(result_root).as_posix()
        for path in result_root.rglob("*")
        if path.is_file()
    }

    assert expected_placeholders <= result_files

    generated_paths = sorted(
        f"examples/results/{path}" for path in result_files - expected_placeholders
    )
    if generated_paths:
        ignored = subprocess.run(
            ["git", "check-ignore", "--no-index", "--stdin"],
            cwd=repository_root,
            input="\n".join(generated_paths),
            check=False,
            capture_output=True,
            text=True,
        )
        assert set(ignored.stdout.splitlines()) == set(generated_paths)
