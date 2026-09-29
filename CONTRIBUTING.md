# Contributing

Use a clean Git worktree and make one reviewable change at a time.

The repository is the long-lived home of the installable `pipls` package. Contributions should
serve package users through the public API, documentation, concise examples, transparent datasets, tests, packaging, or release
maintenance. Paper-specific figure
pipelines, complete publication grids, manuscript tables, and paper-only comparator workflows
belong in downstream reproduction repositories.

Before preparing a substantial change, open an
[issue](https://github.com/stefanlindstroem/pipls/issues) or start a
[discussion](https://github.com/stefanlindstroem/pipls/discussions) so that its scope and
scientific or API implications can be reviewed. Pull requests for substantial changes should link
to that issue or discussion. Small corrections, such as typo and broken-link fixes, may be submitted
directly.

## Development setup

Create and activate a development environment from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
make install
make check
```

For a later terminal session:

```bash
cd /path/to/pipls
source .venv/bin/activate
git status
make check
```

Refresh the editable installation after dependency changes with `make install`.

Run `make help` for the maintained command index. It presents first setup and routine validation
before the task-specific groups.

## Validation commands

After activating the development environment, the primary route is:

```bash
make install
make check
```

`make check` validates the decision registry, runs tests and Ruff, and performs strict mypy
checks. Use the grouped targets when working on a specific part of the repository:

```text
Development
  make decision-check validate the decision registry and retirement map
  make test          run the test suite
  make lint          run Ruff checks
  make format        format Python files with Ruff
  make typecheck     run strict mypy checks
  make clean         remove generated files and caches

Documentation and examples
  make examples      run every numbered example
  make docs                 build the strict MkDocs site and its tutorial figures
  make docs-serve           generate figures and preview the site locally
  make docs-figures         regenerate tutorial figures only
  make docs-static-figures  regenerate committed standalone TeX figures
  make docs-dist            rebuild the site from a clean source distribution

Distribution and maintenance
  make build         build the wheel and source distribution
  make dist-check    verify clean wheel and source-distribution installations
  make snapshot      create an uploadable repository snapshot
```

The `dev` extra owns repository validation, including the maintained example dependencies.
Documentation targets add the dedicated `docs` toolchain. Before documentation work, install
both extras in the editable checkout:

```bash
python -m pip install -e ".[dev,docs]"
```

Run `make examples` when changing executable example behavior, example artifacts, or an
application-facing workflow. It intentionally includes the slower real-data analyses.

Run `make docs-figures` when changing tutorial workflows, tutorial rendering, or plotting behavior
used by generated assets. Standalone documentation diagrams under `docs/assets/figures/` are
committed assets with reproducible TeX sources under `tools/figures/`; regenerate them explicitly
with `make docs-static-figures`, which requires `latexmk` and `pdftocairo`. Ordinary `make docs`
consumes the committed SVG and therefore does not require a TeX installation. Run `make docs` for
public guides, navigation, docstrings, or tutorial assets. `make docs-serve` serves
`http://127.0.0.1:8000/` until stopped with `Ctrl+C`.

Run `make docs-dist` when changing documentation packaging, documentation dependencies,
`MANIFEST.in`, or the source-distribution documentation boundary.

Run `make build` for a quick artifact build when changing packaging, dependencies, included data,
or public modules. Run `make dist-check` before submitting such a change. It builds the artifacts
once, installs the wheel and source distribution into separate clean environments, and exercises
the same installed-package smoke test outside the checkout.

## Compatibility checks

When changing core dependency bounds or compatibility code, also verify the minimum supported
stack in a fresh Python 3.10 environment:

```bash
python -m pip install -c constraints/minimum.txt -e ".[dev]"
make check
```

The constraint file represents the minimum supported dependency lines; it is not an application
lock file. CI separately exercises normal dependency resolution on every supported Python version
and explicit latest-compatible upgrades on Python 3.14.

## Documentation and examples

The root README is a package landing page. Keep repository build, distribution, and maintenance
instructions here in `CONTRIBUTING.md` rather than duplicating them in the README.

Served documentation lives under `docs/`. The strict build checks navigation, internal links,
anchors, mathematical rendering, generated API targets, and tutorial snippets.

Tutorial code excerpts use checked `pymdownx.snippets` sections from maintained numbered examples.
Edit the executable example rather than copying analysis code into Markdown. Tutorial figures under
`docs/assets/generated/` are regenerated by documentation targets and are not committed.

The two tutorial roles are fixed:

- the synthetic tutorial owns the minimum selection and independent-test workflow;
- the Pulp tutorial owns the complete real-data, selection-conditioned OOF, and representative
  interpretation workflow.

Rendering follows one data-first rule: package code computes immutable numerical results, while
numbered examples and tutorial renderers show ordinary Matplotlib construction directly. Keep
`textalloc` optional and confined to annotated biplot label allocation after final axis configuration.
Maintained examples and documentation renderers must remain executable when `textalloc` is absent;
in that case they retain the original Matplotlib text positions. The shared annotation helper may own
that optional label-allocation policy, but chart construction must remain in the example or renderer.
Do not add a replacement plotting submodule or hide chart construction in `examples/_support/`.

The served reference pages own exact API behavior and advanced alternatives. Avoid repeating those
contracts in tutorial prose.

## Data and generated files

Mathematical changes must update the relevant contracts in `docs/maintainers/` and include focused
tests.

Do not commit generated result files without an explicit fixture decision. Generated example
figures, tutorial assets, build artifacts, caches, and archive clutter are ignored. The committed
standalone SVG diagrams under `docs/assets/figures/` are the narrow exception: keep their TeX
sources under `tools/figures/` and regenerate them only through `make docs-static-figures`. Do not
add datasets without verified redistribution and adaptation terms.

## Repository map

- `src/pipls/`: installable package, public API, and canonical reference-data resources;
- `docs/`: served documentation plus excluded maintainer contracts and decision records;
- `examples/`: numbered user workflows and their small support layer;
- `tests/`: numerical, API, integration, documentation, and repository tests;
- `tools/`: documentation and distribution validation helpers;
- `constraints/`: the maintainer-only minimum-dependency environment;
- `docs/maintainers/`: non-served scientific and architectural maintainer contracts.

## Release checklist

Before tagging a release, start from a clean worktree and verify the release metadata, examples,
documentation, and built distributions together:

```bash
git status
make check
make examples
make docs
make docs-dist
make dist-check
```

The release version must agree in `pyproject.toml`, `src/pipls/__init__.py`, `CITATION.cff`, and the
dated release heading in `CHANGELOG.md`. The preferred companion-article citation in
`docs/citation.md` and `CITATION.cff` must also agree. The release-metadata tests enforce the
machine-readable repository-facing identities.

Push the release commit before tagging it and confirm that the GitHub Actions workflows pass and
the GitHub Pages site renders correctly. Then create and push an annotated tag matching the package
version, for example:

```bash
git tag -a v0.1.0 -m "pipls 0.1.0"
git push origin v0.1.0
```

Create the corresponding GitHub Release from that tag. Publishing to a package index is a separate
release action and is not performed by the repository workflows.

## LLM-assisted maintenance

AI assistance is welcome, but every contribution must have a human owner who understands, reviews,
and validates the complete change and remains responsible for it. Disclose material AI assistance
in the pull-request description, including the tool used and the scope of its contribution. Do not
submit AI-generated material that you cannot explain or verify.

Coding assistants should begin with [`AGENTS.md`](AGENTS.md), which routes them to the relevant
human-facing guidance, decisions, and maintainer contracts. Do not create a second set of project
rules solely for a particular assistant or tool.
