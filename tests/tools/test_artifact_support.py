from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from tools._artifact_support import (
    clean_subprocess_environment,
    safe_extract_sdist,
    single_artifact,
    source_distribution_environment,
)


def _write_tar(archive_path: Path, members: dict[str, bytes]) -> None:
    with tarfile.open(archive_path, mode="w:gz") as archive:
        for name, content in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))


def test_clean_subprocess_environment_removes_python_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONHOME", "/tmp/python-home")
    monkeypatch.setenv("PYTHONPATH", "/tmp/python-path")
    monkeypatch.setenv("PIP_CACHE_DIR", "/tmp/pip-cache")

    environment = clean_subprocess_environment()

    assert "PYTHONHOME" not in environment
    assert "PYTHONPATH" not in environment
    assert environment["OMP_NUM_THREADS"] == "1"
    assert environment["OPENBLAS_NUM_THREADS"] == "1"
    assert environment["PIP_CACHE_DIR"] == "/tmp/pip-cache"


def test_source_distribution_environment_uses_extracted_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("PYTHONPATH", "/tmp/development-checkout")
    source = tmp_path / "pipls-1.2.3"

    environment = source_distribution_environment(source)

    assert environment["PYTHONPATH"] == str(source / "src")
    assert environment["PYTHONPATH"] != "/tmp/development-checkout"


def test_single_artifact_requires_exactly_one_match(tmp_path: Path) -> None:
    artifact = tmp_path / "pipls-1.2.3.tar.gz"
    artifact.touch()

    assert single_artifact(tmp_path, "*.tar.gz", "source distribution") == artifact

    (tmp_path / "pipls-0.0.1.tar.gz").touch()
    with pytest.raises(RuntimeError, match="Expected one source distribution"):
        single_artifact(tmp_path, "*.tar.gz", "source distribution")


def test_safe_extract_sdist_returns_the_single_root(tmp_path: Path) -> None:
    artifact = tmp_path / "pipls.tar.gz"
    _write_tar(
        artifact,
        {
            "pipls-1.2.3/README.md": b"Pi-PLS\n",
            "pipls-1.2.3/src/pipls/__init__.py": b"",
        },
    )

    source = safe_extract_sdist(artifact, tmp_path / "extracted")

    assert source == tmp_path / "extracted" / "pipls-1.2.3"
    assert (source / "README.md").read_bytes() == b"Pi-PLS\n"


def test_safe_extract_sdist_rejects_path_traversal(tmp_path: Path) -> None:
    artifact = tmp_path / "pipls.tar.gz"
    _write_tar(artifact, {"pipls-1.2.3/../../escaped.txt": b"bad"})

    with pytest.raises(RuntimeError, match="Unsafe source-distribution member"):
        safe_extract_sdist(artifact, tmp_path / "extracted")
