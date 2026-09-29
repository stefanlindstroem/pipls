# PiPLS repository instructions

This file is the entry point for coding assistants. Human maintainers retain authority over
scientific, public-API, licensing, and release decisions. AI-assisted work must be reviewed and
validated by a human before inclusion.

## Start here

1. Read `CONTRIBUTING.md` for setup, validation, documentation, and release workflows.
2. Inspect the affected source, tests, and public documentation before proposing a change.
3. Use `docs/maintainers/index.md` to find only the decision records and scientific contracts that
   are relevant to the task; do not load the entire maintenance history by default.
4. Check `git status` and preserve unrelated or user-authored changes.

Source and tests are evidence of implemented behavior. Public documentation describes the supported
user experience. Numbered decisions explain accepted design choices. If these disagree, surface the
conflict instead of silently choosing one.

## Repository boundaries

- `src/pipls/` is the installable package and uses the standard Python `src` layout.
- `docs/` contains self-contained user documentation; `docs/decisions/` contains non-served
  maintainer records.
- `examples/` contains maintained user workflows, while `tools/` contains repository automation.
- Paper-specific reproduction pipelines and manuscript-only comparisons belong in downstream
  repositories that depend on a tagged PiPLS release.

Keep every rule in one canonical location. Link to an existing contract instead of restating it in
slightly different words. Do not encode scientific or API requirements only in assistant-specific
instructions.

## Changes and validation

Make one coherent, reviewable change at a time. Preserve public imports and behavior unless the
requested work explicitly changes them. Add focused tests for behavioral changes and update the
canonical user documentation in the same change.

Use the validation commands and applicability rules in `CONTRIBUTING.md`. At minimum, run
`make check`; also run the documentation, example, static-figure, and distribution targets relevant
to the files changed. Report checks as passed, failed, or not run, and never describe inspection as
validation.
