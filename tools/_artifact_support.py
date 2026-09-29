"""Private helpers shared by artifact-validation entry points."""

from __future__ import annotations

import os
import subprocess
import sys
import tarfile
from pathlib import Path


def run(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    """Run one checked subprocess while showing the invoked command."""
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)

def clean_subprocess_environment() -> dict[str, str]:
    """Return an environment isolated from the current Python checkout."""
    environment = os.environ.copy()
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    return environment

def source_distribution_environment(source: Path) -> dict[str, str]:
    """Return a clean environment that imports from an extracted sdist."""
    environment = clean_subprocess_environment()
    environment["PYTHONPATH"] = str(source / "src")
    return environment

def venv_python(environment: Path) -> Path:
    """Return the Python executable inside a virtual environment."""
    if os.name == "nt":
        return environment / "Scripts" / "python.exe"
    return environment / "bin" / "python"

def single_artifact(artifacts: Path, pattern: str, label: str) -> Path:
    """Return the only matching artifact or raise a useful error."""
    matches = sorted(artifacts.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {label}, found {len(matches)}.")
    return matches[0]


def safe_extract_sdist(artifact: Path, destination: Path) -> Path:
    """Extract one source distribution without accepting links or traversal."""
    destination.mkdir(parents=True, exist_ok=False)
    resolved_destination = destination.resolve()

    with tarfile.open(artifact, mode="r:gz") as archive:
        members = archive.getmembers()
        roots = {
            Path(member.name).parts[0]
            for member in members
            if Path(member.name).parts
        }
        if len(roots) != 1:
            raise RuntimeError(
                "The source distribution must contain one top-level directory."
            )

        for member in members:
            member_path = Path(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise RuntimeError(
                    f"Unsafe source-distribution member: {member.name}"
                )
            if member.issym() or member.islnk():
                raise RuntimeError(
                    f"Source-distribution links are unsupported: {member.name}"
                )
            target = (destination / member_path).resolve()
            if not target.is_relative_to(resolved_destination):
                raise RuntimeError(
                    f"Source-distribution member escapes extraction: {member.name}"
                )

        if sys.version_info >= (3, 12):
            archive.extractall(destination, filter="data")
        else:
            archive.extractall(destination)

    return destination / roots.pop()
