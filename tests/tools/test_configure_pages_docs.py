from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run_configure_pages_docs(
    *,
    output: Path,
    repository: str,
    server_url: str = "https://github.com",
    branch: str = "main",
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(_repository_root() / "tools" / "configure_pages_docs.py"),
            "--repository",
            repository,
            "--server-url",
            server_url,
            "--branch",
            branch,
            "--output",
            str(output),
        ],
        cwd=_repository_root(),
        capture_output=True,
        text=True,
        check=False,
    )


def test_configure_pages_docs_writes_project_site_overlay(tmp_path: Path) -> None:
    output = tmp_path / "pages" / "mkdocs.yml"
    output.parent.mkdir()

    completed = _run_configure_pages_docs(
        output=output,
        repository="openai/pipls",
        branch="trunk",
    )

    assert completed.returncode == 0, completed.stderr
    config = yaml.safe_load(output.read_text(encoding="utf-8"))
    expected_inherit = Path(
        os.path.relpath(
            (_repository_root() / "mkdocs.yml").resolve(),
            start=output.parent.resolve(),
        )
    ).as_posix()
    assert config == {
        "INHERIT": expected_inherit,
        "site_url": "https://openai.github.io/pipls/",
        "repo_url": "https://github.com/openai/pipls",
        "edit_uri": "edit/trunk/docs/",
    }


def test_configure_pages_docs_writes_user_site_url(tmp_path: Path) -> None:
    output = tmp_path / "mkdocs.yml"

    completed = _run_configure_pages_docs(
        output=output,
        repository="openai/openai.github.io",
    )

    assert completed.returncode == 0, completed.stderr
    config = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert config["site_url"] == "https://openai.github.io/"


@pytest.mark.parametrize(
    ("repository", "server_url", "branch"),
    [
        ("invalid", "https://github.com", "main"),
        ("openai/pipls", "http://github.com", "main"),
        ("openai/pipls", "https://github.com", ""),
    ],
)
def test_configure_pages_docs_rejects_invalid_inputs(
    tmp_path: Path,
    repository: str,
    server_url: str,
    branch: str,
) -> None:
    output = tmp_path / "mkdocs.yml"

    completed = _run_configure_pages_docs(
        output=output,
        repository=repository,
        server_url=server_url,
        branch=branch,
    )

    assert completed.returncode != 0
    assert not output.exists()
