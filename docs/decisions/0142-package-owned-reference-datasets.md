# Decision 0142: package-owned reference datasets

## Status

Accepted; implemented.

## Context

Pulp was the first named dataset distributed with Pi-PLS. The resulting
`load_pulp()` workflow is useful in an installed package, preserves labels and metadata,
and avoids a repository-relative file dependency. Sugarcane and Tobacco now have the same relevant
properties: they are licensed, curated reference analyses with fixed predictor and response
matrices, documented scientific transformations, and maintained Pi-PLS examples.

Keeping Sugarcane and Tobacco only under `datasets/` creates two avoidable inconsistencies. Their
examples require a repository checkout and pandas merely to read fixed CSV resources, while Pulp
works from an installed wheel. Their analysis matrices are also less discoverable through the
Python API even though they are recurring reference datasets with verified redistribution rights.

Moving the matrices into package resources must not make them Python-only or obscure them from R,
C++, MATLAB, Julia, or other programming users. The package resources are ordinary text files in a
wheel, source distribution, or source checkout. Their canonical locations and accompanying
metadata, README, and license files therefore need an explicit language-neutral documentation
contract.

The maintained package keeps one active matrix copy per dataset and no compatibility aliases,
generic registry, or duplicate repository layout.

## Decision

Pi-PLS owns three named reference-dataset loaders in `pipls.datasets`:

```python
def load_pulp(
    *,
    return_X_y: bool = False,
) -> PiPLSDataset | tuple[FloatArray, FloatArray]:
    ...


def load_sugarcane(
    *,
    return_X_y: bool = False,
) -> PiPLSDataset | tuple[FloatArray, FloatArray]:
    ...


def load_tobacco(
    *,
    return_X_y: bool = False,
) -> PiPLSDataset | tuple[FloatArray, FloatArray]:
    ...
```

All three loaders share one narrow contract:

- the default result is a `PiPLSDataset` with fresh read-only `X` and `Y` arrays;
- `return_X_y=True` returns fresh read-only `float64` predictor and response arrays;
- loaders are exported from `pipls.datasets`, not from the top-level `pipls` namespace;
- loading performs no network access, filtering, imputation, centering, scaling, spectral
  preprocessing, or model fitting;
- runtime parsing uses `importlib.resources` plus standard-library CSV and JSON support;
- no pandas, PyYAML, optional data extra, `as_frame` mode, downloader, registry, or generic
  `load_dataset(name)` interface is introduced.

The canonical active resources are:

```text
src/pipls/_data/
    pulp/
        X.csv
        Y.csv
        metadata.json
        README.md
        LICENSE.txt
    sugarcane/
        X.csv
        Y.csv
        metadata.json
        README.md
        LICENSE.txt
    tobacco/
        X.csv
        Y.csv
        metadata.json
        README.md
        LICENSE.txt
```

Each `X.csv` and `Y.csv` pair is the exact analysis-facing matrix pair returned by its loader.
Metadata records ordered labels, source-level transformations, row alignment, public provenance,
licensing, raw-resource hashes, and canonical `float64` array hashes. Source sample numbers,
exclusions, and row-alignment information remain documentary metadata where they are available;
the public dataset container does not manufacture package-level sample identifiers.

The resource directories are intentionally language-neutral. Public dataset documentation must
list the canonical paths and explain that the same CSV, JSON, README, and license files are
available:

- in a source checkout or unpacked source distribution under `src/pipls/_data/<dataset>/`;
- inside a wheel, which is a ZIP archive, under `pipls/_data/<dataset>/`;
- in an installed environment under the environment-specific
  `<site-packages>/pipls/_data/<dataset>/` location.

For reproducible non-Python use, documentation should recommend a tagged source release, source
distribution, or wheel rather than a moving development branch or an environment-specific installed
path. Each resource README states that the files may be used independently of Python and identifies
the local matrix, metadata, README, and license resources.

There is one active matrix representation for each named dataset. Pulp, Sugarcane, and Tobacco
each have one `X.csv`/`Y.csv` pair under `src/pipls/_data/<dataset>/`, accompanied by
`metadata.json`, `README.md`, and `LICENSE.txt`. Former repository-level Sugarcane and Tobacco
matrix copies are removed rather than retained as duplicate active data. The hidden historical Pulp
archive remains excluded development history and is not an active data source.

All three loaders share one private package-resource pipeline for CSV and JSON parsing, header and
dimensional validation, and metadata. Dedicated repository tests verify resource and
canonical-array integrity, while distribution qualification verifies installed loading; runtime
loaders do not rehash package resources on every call. Every maintained reference-data consumer
uses its named loader. Wheels and source
distributions include all five files for each dataset, isolated installations load all three
datasets, and tests protect one active matrix location.

This decision generalizes the former Pulp-only loader to a closed set of three maintained reference
datasets. Arbitrary user-owned data remains ordinary array-like `X` and `Y`; no generic registry,
downloader, repository-only layout, compatibility alias, pandas return mode, or hidden preprocessing
layer is introduced.

## Consequences

- Pulp, Sugarcane, and Tobacco can be loaded from a clean installed package through one consistent
  named-loader vocabulary.
- Their canonical CSV matrices remain directly usable outside Python and are documented as such.
- Examples 03, 05, and 06 use the named package loaders rather than repository-relative paths or
  pandas for fixed dataset ingestion.
- Package distributions grow because the spectral matrices travel with the base installation.
- Resource hashes, canonical-array hashes, installed-distribution loading, and single-copy location
  become durable tests for all three datasets.
- The package still owns no general real-data ingestion framework: arbitrary users continue to
  supply `X` and `Y` directly.
- No generic registry, downloader, compatibility alias, pandas return mode, hidden preprocessing,
  or duplicate active matrix representation is introduced.
