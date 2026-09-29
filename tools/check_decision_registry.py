#!/usr/bin/env python3
"""Validate the maintained decision registry and retirement map."""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from urllib.parse import unquote

_DECISION_FILE_RE = re.compile(r"^(?P<number>\d{4})-[a-z0-9][a-z0-9-]*\.md$")
_MARKDOWN_LINK_RE = re.compile(r"\[[^\]]*\]\((?P<target>[^)]+)\)")
_RETIRED_ROW_RE = re.compile(r"^\|\s*`(?P<filename>\d{4}-[^`]+\.md)`\s*\|")
_DECISION_REFERENCE_RE = re.compile(
    r"\bDecisions?\s+(?P<numbers>"
    r"\d{4}(?:(?:\s*,\s*and\s+|\s*,\s*|\s+and\s+)\d{4})*)\b"
)
# The maintained tree inherited these two collisions before registry normalization. Their exact
# current/retired filename pairs are frozen; no additional decision-number reuse is permitted.
_LEGACY_NUMBER_COLLISIONS = {
    "0153": (
        "0153-independent-block-scaling-controls.md",
        "0153-remove-package-owned-leave-one-out-support.md",
    ),
    "0154": (
        "0154-full-domain-predictor-rank-selection.md",
        "0154-behavioral-test-suite-cleanup.md",
    ),
}


def _decision_number(filename: str) -> str:
    match = _DECISION_FILE_RE.fullmatch(filename)
    if match is None:
        raise ValueError(f"not a numbered decision filename: {filename}")
    return match.group("number")


def _numbered_markdown_links(path: Path) -> list[str]:
    links: list[str] = []
    for match in _MARKDOWN_LINK_RE.finditer(path.read_text(encoding="utf-8")):
        target = match.group("target").strip().strip("<>")
        target = unquote(target.split("#", 1)[0].split("?", 1)[0])
        filename = Path(target).name
        if _DECISION_FILE_RE.fullmatch(filename):
            links.append(filename)
    return links


def _check_local_markdown_links(paths: list[Path]) -> list[str]:
    errors: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for match in _MARKDOWN_LINK_RE.finditer(text):
            target = match.group("target").strip().strip("<>")
            if "://" in target or target.startswith("mailto:"):
                continue
            target_without_fragment = target.split("#", 1)[0].split("?", 1)[0]
            if not target_without_fragment:
                continue
            resolved = (path.parent / unquote(target_without_fragment)).resolve()
            if not resolved.exists():
                errors.append(
                    f"{path}: unresolved local Markdown link {match.group('target')!r}"
                )
    return errors


def check_registry(root: Path) -> list[str]:
    """Return structural decision-registry errors for *root*."""
    decision_dir = root / "docs" / "decisions"
    index_path = decision_dir / "index.md"
    retirements_path = decision_dir / "retirements.md"
    required = (decision_dir, index_path, retirements_path)
    missing_required = [path for path in required if not path.exists()]
    if missing_required:
        return [f"missing required registry path: {path}" for path in missing_required]

    current_paths = sorted(
        path
        for path in decision_dir.iterdir()
        if path.is_file() and _DECISION_FILE_RE.fullmatch(path.name)
    )
    current_filenames = [path.name for path in current_paths]
    current_numbers = [_decision_number(filename) for filename in current_filenames]
    current_by_number: dict[str, str] = dict(zip(current_numbers, current_filenames, strict=True))

    errors: list[str] = []
    duplicate_current_numbers = sorted(
        number for number, count in Counter(current_numbers).items() if count != 1
    )
    if duplicate_current_numbers:
        errors.append(
            "multiple current decision files use number(s): "
            + ", ".join(duplicate_current_numbers)
        )

    index_links = _numbered_markdown_links(index_path)
    index_counts = Counter(index_links)
    missing_from_index = sorted(set(current_filenames) - set(index_links))
    unknown_in_index = sorted(set(index_links) - set(current_filenames))
    duplicate_index_entries = sorted(
        filename for filename, count in index_counts.items() if count != 1
    )
    if missing_from_index:
        errors.append("current decisions missing from index: " + ", ".join(missing_from_index))
    if unknown_in_index:
        errors.append("index references non-current decisions: " + ", ".join(unknown_in_index))
    if duplicate_index_entries:
        errors.append("index repeats decision links: " + ", ".join(duplicate_index_entries))

    retired_filenames = [
        match.group("filename")
        for line in retirements_path.read_text(encoding="utf-8").splitlines()
        if (match := _RETIRED_ROW_RE.match(line)) is not None
    ]
    retired_counts = Counter(retired_filenames)
    duplicate_retirements = sorted(
        filename for filename, count in retired_counts.items() if count != 1
    )
    if duplicate_retirements:
        errors.append(
            "retirement map repeats filenames: " + ", ".join(duplicate_retirements)
        )

    current_retired_filename_overlap = sorted(set(current_filenames) & set(retired_filenames))
    if current_retired_filename_overlap:
        errors.append(
            "filenames cannot be both current and retired: "
            + ", ".join(current_retired_filename_overlap)
        )

    retired_by_number: dict[str, list[str]] = {}
    for filename in retired_filenames:
        number = _decision_number(filename)
        retired_by_number.setdefault(number, []).append(filename)

    number_overlap = sorted(set(current_by_number) & set(retired_by_number))
    unexpected_overlap = sorted(set(number_overlap) - set(_LEGACY_NUMBER_COLLISIONS))
    if unexpected_overlap:
        errors.append(
            "decision numbers are reused outside the frozen legacy exceptions: "
            + ", ".join(unexpected_overlap)
        )

    for number in sorted(set(number_overlap) & set(_LEGACY_NUMBER_COLLISIONS)):
        expected_current, expected_retired = _LEGACY_NUMBER_COLLISIONS[number]
        actual_current = current_by_number[number]
        actual_retired = retired_by_number[number]
        if actual_current != expected_current or actual_retired != [expected_retired]:
            errors.append(
                f"legacy collision {number} changed: current={actual_current!r}, "
                f"retired={actual_retired!r}"
            )

    maintainer_paths = sorted((root / "docs" / "maintainers").rglob("*.md"))
    active_reference_paths = current_paths + maintainer_paths
    for path in active_reference_paths:
        for match in _DECISION_REFERENCE_RE.finditer(path.read_text(encoding="utf-8")):
            for number in re.findall(r"\d{4}", match.group("numbers")):
                if number not in current_by_number:
                    errors.append(f"{path}: active reference to non-current Decision {number}")

    decision_markdown_paths = sorted(decision_dir.glob("*.md"))
    errors.extend(_check_local_markdown_links(decision_markdown_paths + maintainer_paths))
    return errors


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root (default: inferred from this script)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = args.root.resolve()
    errors = check_registry(root)
    if errors:
        for error in errors:
            print(f"decision registry error: {error}")
        return 1

    decision_dir = root / "docs" / "decisions"
    current_count = sum(
        1
        for path in decision_dir.iterdir()
        if path.is_file() and _DECISION_FILE_RE.fullmatch(path.name)
    )
    retired_count = sum(
        1
        for line in (decision_dir / "retirements.md").read_text(encoding="utf-8").splitlines()
        if _RETIRED_ROW_RE.match(line) is not None
    )
    print(
        "Decision registry is consistent: "
        f"{current_count} current, {retired_count} retired, "
        f"{len(_LEGACY_NUMBER_COLLISIONS)} frozen legacy number collisions."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
