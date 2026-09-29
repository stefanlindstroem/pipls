from __future__ import annotations

import argparse
import os
from pathlib import Path
from urllib.parse import urlparse

import yaml


def _https_url(value: str, *, name: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError(f"{name} must be an absolute HTTPS URL")
    return value.rstrip("/")


def _pages_site_url(repository: str) -> str:
    try:
        owner, name = repository.split("/", 1)
    except ValueError as error:
        raise ValueError("repository must have the form 'owner/name'") from error
    if not owner or not name:
        raise ValueError("repository must have the form 'owner/name'")
    if name.casefold() == f"{owner}.github.io".casefold():
        return f"https://{owner}.github.io/"
    return f"https://{owner}.github.io/{name}/"


def _branch_name(value: str) -> str:
    if not value or value != value.strip() or "\n" in value or "\r" in value:
        raise ValueError("branch must be a non-empty branch name")
    return value


def _inherit_path(*, source: Path, output: Path) -> str:
    if not source.is_file():
        raise ValueError(f"MkDocs configuration does not exist: {source}")
    relative = os.path.relpath(source.resolve(), start=output.parent.resolve())
    return Path(relative).as_posix()


def write_pages_config(
    *,
    source: Path,
    output: Path,
    repository: str,
    server_url: str,
    branch: str,
) -> None:
    repository_url = f"{_https_url(server_url, name='server_url')}/{repository}"
    config = {
        "INHERIT": _inherit_path(source=source, output=output),
        "site_url": _pages_site_url(repository),
        "repo_url": repository_url,
        "edit_uri": f"edit/{_branch_name(branch)}/docs/",
    }
    output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the GitHub Pages MkDocs configuration.")
    parser.add_argument("--repository", required=True)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[1]
    write_pages_config(
        source=repository_root / "mkdocs.yml",
        output=arguments.output,
        repository=arguments.repository,
        server_url=arguments.server_url,
        branch=arguments.branch,
    )


if __name__ == "__main__":
    main()
