# Decision 0103: installation and optional-dependency boundary

## Status

Accepted.

## Context

The package exposed a `data` extra although the runtime provides no dataset registry or loader, and
that extra had no maintained consumer. The development extra retained `pytest-cov` although no
maintained target generated coverage. The documentation extra retained Ruff although documentation
targets do not invoke linting. Public source-checkout instructions also used editable installation,
which is a contributor workflow rather than the ordinary user route.

## Decision

Expose only three optional dependency groups:

- `examples` owns pandas, Matplotlib, and optional `textalloc` for numbered examples;
- `docs` owns source-distribution construction, tutorial rendering, and the strict MkDocs toolchain;
- `dev` owns complete repository validation and therefore includes the example dependencies,
  PyYAML, Python-3.10 TOML support, pytest, mypy, Ruff, and artifact construction.

Remove the unused `data` extra, `pytest-cov`, and Ruff from the documentation extra. Shipped Pulp,
Sugarcane, and Tobacco data remain ordinary repository CSV files and require no package-specific
installation extra.

Public README and served-documentation commands use ordinary noneditable installation from a source
checkout. Editable installation remains the contributor and CI workflow documented in
`CONTRIBUTING.md`.

Optional rendering helpers remain non-fatal when the optional package itself is absent. In
particular, maintained Pulp biplot rendering must remain executable with Matplotlib alone and use
its plain endpoint-label path when `textalloc` is unavailable. An unrelated import failure from an
installed optional package must propagate rather than be disguised as dependency absence. No
graphics dependency enters the runtime `pipls` import surface.

## Consequences

The optional dependency surface now describes maintained executable workflows rather than possible
future uses. Dataset access is not implied by installation metadata, and new users are not directed
toward editable package state. Contributors retain one complete `dev` environment, while strict
documentation work adds `docs` explicitly.
