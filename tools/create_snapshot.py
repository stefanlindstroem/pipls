#!/usr/bin/env python3
"""Create a portable, root-relative snapshot of a clean committed tree."""

from __future__ import annotations

import gzip
import io
import os
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

_ARCHIVE_MTIME = 1_577_836_800  # 2020-01-01 00:00:00 UTC
_CACHE_DIRECTORIES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
}
_GENERATED_PREFIXES = (
    "build/",
    "dist/",
    "docs/_build/",
    "docs/assets/generated/",
    "htmlcov/",
    "site/",
)


def _run_git(root: Path | None, *arguments: str, text: bool = True) -> str | bytes:
    command = ["git"]
    if root is not None:
        command.extend(("-C", str(root)))
    command.extend(arguments)
    completed = subprocess.run(command, check=True, capture_output=True, text=text)
    return completed.stdout


def _repository_root() -> Path:
    return Path(str(_run_git(None, "rev-parse", "--show-toplevel")).strip()).resolve()


def _is_prohibited_artifact(path: str) -> bool:
    parts = PurePosixPath(path).parts
    if any(part in _CACHE_DIRECTORIES or part.endswith(".egg-info") for part in parts):
        return True
    if path.startswith(_GENERATED_PREFIXES):
        return True
    if path.endswith((".pyc", ".pyo")):
        return True
    if PurePosixPath(path).name in {".coverage", "coverage.xml"}:
        return True
    if PurePosixPath(path).name.startswith(".coverage."):
        return True
    return path.startswith("examples/results/") and PurePosixPath(path).name != ".gitkeep"


def _package_version(root: Path) -> str:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    completed = subprocess.run(
        [sys.executable, "-c", "import pipls; print(pipls.__version__)"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def _write_archive(source: Path, output: Path) -> None:
    with output.open("wb") as raw_output:
        with gzip.GzipFile(fileobj=raw_output, mode="wb", filename="", mtime=_ARCHIVE_MTIME) as gz:
            with tarfile.open(fileobj=gz, mode="w", format=tarfile.GNU_FORMAT) as archive:
                paths = sorted(
                    source.rglob("*"),
                    key=lambda item: item.relative_to(source).as_posix(),
                )
                for path in paths:
                    relative = path.relative_to(source).as_posix()
                    info = archive.gettarinfo(str(path), arcname=relative)
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = _ARCHIVE_MTIME
                    if info.isfile():
                        with path.open("rb") as stream:
                            archive.addfile(info, stream)
                    else:
                        archive.addfile(info)


def create_snapshot(output_argument: str | None = None) -> Path:
    """Create and return a snapshot archive path."""

    root = _repository_root()
    output = (
        Path(output_argument).expanduser()
        if output_argument is not None
        else root.parent / f"{root.name}-snapshot.tar.gz"
    )
    if not output.is_absolute():
        output = Path.cwd() / output
    output = output.resolve()

    status = str(
        _run_git(
            root,
            "status",
            "--porcelain=v1",
            "--untracked-files=normal",
            "--ignore-submodules=none",
        )
    )
    if status:
        raise RuntimeError(f"Refusing to create a snapshot from a dirty worktree:\n{status}")

    tracked = bytes(_run_git(root, "ls-tree", "-r", "-z", "--name-only", "HEAD", text=False))
    prohibited = sorted(
        path.decode("utf-8")
        for path in tracked.split(b"\0")
        if path and _is_prohibited_artifact(path.decode("utf-8"))
    )
    if prohibited:
        listing = "\n".join(f"  {path}" for path in prohibited)
        raise RuntimeError(
            "Refusing to create a snapshot with tracked cache or generated artifacts:\n" + listing
        )

    commit = str(_run_git(root, "rev-parse", "HEAD")).strip()
    branch_result = subprocess.run(
        ["git", "-C", str(root), "symbolic-ref", "--quiet", "--short", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    branch = branch_result.stdout.strip() if branch_result.returncode == 0 else "detached"
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    metadata = (
        f"project: {root.name}\n"
        f"commit: {commit}\n"
        f"branch: {branch}\n"
        "dirty: false\n"
        f"created_utc: {created}\n"
        f"python: {sys.version.split()[0]}\n"
        f"package_version: {_package_version(root)}\n"
    )

    committed_archive = bytes(_run_git(root, "archive", "--format=tar", "HEAD", text=False))
    with tempfile.TemporaryDirectory() as temporary_directory:
        staging = Path(temporary_directory)
        with tarfile.open(fileobj=io.BytesIO(committed_archive), mode="r:") as archive:
            archive.extractall(staging)
        (staging / "SNAPSHOT_INFO").write_text(metadata, encoding="utf-8")
        _write_archive(staging, output)

    return output


def main() -> int:
    try:
        output = create_snapshot(sys.argv[1] if len(sys.argv) > 1 else None)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(error, file=sys.stderr)
        return 1
    print(f"Created {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
