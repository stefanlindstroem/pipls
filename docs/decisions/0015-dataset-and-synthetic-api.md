# Decision 0015: dataset container and synthetic generator API

## Status

Accepted in Phase E1; simplified to the single synthetic generator.

## Context

The package needs one stable in-memory boundary for packaged datasets and structured experiments.
The boundary must preserve matrix labels and dataset metadata without making preprocessing
decisions or exposing mutable arrays. Synthetic validation and examples separately need
deterministic predictor-specific, shared, and response-specific latent structure without changing
NumPy's global random state.

The synthetic facility is support infrastructure for testing, validation, examples, and
reproducibility. It is not part of the Pi-PLS estimator or fitting algorithm, so the public surface
should contain no more generator machinery than those uses require.

## Decision

- Public dataset functionality lives in `pipls.datasets`, following the scikit-learn convention of
  a dedicated dataset namespace rather than expanding the package root.
- `PiPLSDataset` is the common validated container for packaged datasets and structured
  experiments. It stores `X`, `Y`, feature names, target names, and metadata; `X` and `Y` are
  defensive read-only `float64` copies.
- `Y` is normalized to a two-dimensional array, including single-response datasets.
- Reference-dataset provenance is retained once inside `metadata`; it is not duplicated as a
  separate container field and is not required for user-constructed datasets.
- The top-level metadata mapping is shallow-copied. Metadata
  values are documentary user/package data and are not recursively normalized, copied, or frozen.
- `make_synthetic_data()` is the single public synthetic generator. It uses a local seeded
  `numpy.random.Generator` and returns the generated `(X, Y)` arrays directly.
- Exact internal random-draw order is an implementation detail rather than public API. A fixed seed
  and identical arguments reproduce the same arrays for a given package implementation.
- The generator accepts predictor-specific, shared, and response-specific latent dimensions,
  predictor/response noise levels, and one deterministic unsigned 32-bit seed. Zero latent ranks
  and zero noise are valid.
- Train/test partitioning is not generator API. Generate one reproducible `(X, Y)` pair and split
  rows explicitly when separate analysis blocks are needed.
- Real dataset loading, implicit or explicit downloading, conversion, checksums, and registry
  resolution are not part of the synthetic generator contract.

## Consequences

- Synthetic estimator tests and examples use one package-owned generator rather than parallel
  synthetic frameworks.
- Train/test demonstrations use ordinary row splitting without a separate generator abstraction.
- Real-data examples may use plain arrays or data frames and are not required to construct this
  container.
- Dataset migrations must keep their reading and matrix-construction steps explicit rather than
  adding a generic runtime loader.
