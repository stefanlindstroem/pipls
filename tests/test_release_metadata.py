from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml

import pipls

if importlib.util.find_spec("tomllib") is not None:
    import tomllib
else:  # pragma: no cover - exercised on Python 3.10 in CI
    import tomli as tomllib


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_REPOSITORY = "https://github.com/stefanlindstroem/pipls"
EXPECTED_DOCUMENTATION = "https://stefanlindstroem.github.io/pipls/"
EXPECTED_RELEASE_DATE = "2026-09-29"
EXPECTED_SOFTWARE_AUTHORS = ["Vishal Agrawal", "Stefan B. Lindström"]
EXPECTED_ARTICLE_AUTHORS = [
    "Vishal Agrawal",
    "Fritjof Nilsson",
    "Stefan B. Lindström",
]


def _project_metadata() -> dict[str, object]:
    with (REPOSITORY_ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["project"]


def _citation_metadata() -> dict[str, object]:
    return yaml.safe_load((REPOSITORY_ROOT / "CITATION.cff").read_text(encoding="utf-8"))


def test_release_version_is_consistent_across_public_metadata() -> None:
    project = _project_metadata()
    citation = _citation_metadata()
    version = str(project["version"])
    assert version != "0.0.0"
    assert pipls.__version__ == version
    assert citation["version"] == version
    assert str(citation["date-released"]) == EXPECTED_RELEASE_DATE
    changelog = (REPOSITORY_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert f"## {version} - {EXPECTED_RELEASE_DATE}" in changelog


def test_release_urls_are_consistent_across_public_metadata() -> None:
    project = _project_metadata()
    citation = _citation_metadata()
    project_urls = project["urls"]
    assert isinstance(project_urls, dict)

    assert project_urls["Source"] == EXPECTED_REPOSITORY
    assert project_urls["Issues"] == f"{EXPECTED_REPOSITORY}/issues"
    assert project_urls["Homepage"] == EXPECTED_DOCUMENTATION
    assert project_urls["Documentation"] == EXPECTED_DOCUMENTATION
    assert project_urls["Changelog"] == f"{EXPECTED_REPOSITORY}/blob/main/CHANGELOG.md"
    assert citation["repository-code"] == EXPECTED_REPOSITORY
    assert citation["url"] == EXPECTED_DOCUMENTATION


def test_software_and_article_authors_are_distinct() -> None:
    project = _project_metadata()
    citation = _citation_metadata()

    project_authors = [str(author["name"]) for author in project["authors"]]
    citation_authors = [
        f"{author['given-names']} {author['family-names']}" for author in citation["authors"]
    ]
    article_authors = [
        f"{author['given-names']} {author['family-names']}"
        for author in citation["preferred-citation"]["authors"]
    ]

    assert project_authors == EXPECTED_SOFTWARE_AUTHORS
    assert citation_authors == EXPECTED_SOFTWARE_AUTHORS
    assert article_authors == EXPECTED_ARTICLE_AUTHORS


def test_annotation_layout_extra_is_textalloc_only() -> None:
    project = _project_metadata()
    optional = project["optional-dependencies"]
    assert isinstance(optional, dict)

    for group in ("dev", "examples", "docs"):
        requirements = [str(requirement) for requirement in optional[group]]
        assert any(requirement.startswith("textalloc>=") for requirement in requirements)
        assert all(not requirement.lower().startswith("adjusttext") for requirement in requirements)


def test_documentation_workflow_uses_least_privilege_pages_permissions() -> None:
    workflow = yaml.safe_load(
        (REPOSITORY_ROOT / ".github" / "workflows" / "documentation.yml").read_text(
            encoding="utf-8"
        )
    )

    assert workflow["permissions"] == {"contents": "read", "pages": "read"}
    assert workflow["jobs"]["deploy"]["permissions"] == {
        "pages": "write",
        "id-token": "write",
    }
