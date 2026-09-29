# Maintainer reference

This directory contains non-served scientific and architectural contracts that are useful when
changing PiPLS. It is not a second user guide and it is not specific to one coding assistant.
Coding assistants enter through the repository-level [`AGENTS.md`](../../AGENTS.md); human
contributors enter through [`CONTRIBUTING.md`](../../CONTRIBUTING.md).

Human maintainers retain authority over scientific, public-API, licensing, and release decisions.
Source and tests show implemented behavior, public documentation explains supported use, and
numbered decisions record accepted design choices. A conflict among them must be surfaced and
resolved explicitly.

## Canonical owners

| Information | Canonical location |
|---|---|
| User-facing model and API guidance | `docs/` and public source docstrings |
| Executable behavior and numerical invariants | `src/pipls/` and `tests/` |
| Contributor setup and validation | `CONTRIBUTING.md` and `Makefile` |
| Accepted design decisions | `docs/decisions/index.md` and numbered records |
| Release history | `CHANGELOG.md` |
| Coding-assistant routing | `AGENTS.md` |
| Detailed non-served contracts | this directory |

Do not restate a contract merely to make another document self-contained. Link to its canonical
owner when the intended audience can follow the link; summarize only when a public guide needs a
short, independently understandable explanation.

## Scientific contracts

Only detail that does not have a clearer canonical owner is retained here:

- `theory.md` gives the complete conceptual derivation, interpretation, and limiting cases.
- `mathematics.md` records normative equations, dimensions, notation, and invariants.
- `numerical_contracts.md` records numerical algorithms, tolerances, degeneracy handling, and
  comparison rules.

Public interfaces belong to source docstrings and the served API reference. Dataset policy,
analysis ownership, repository boundaries, and accepted exclusions belong to their numbered
decisions and public guides. Current work belongs in issues or discussions rather than a second
repository roadmap. Implementation and validation workflow belongs to `CONTRIBUTING.md`.

The complete maintenance workflow, validation matrix, and optional snapshot handoff are defined in
`CONTRIBUTING.md` rather than repeated here.
