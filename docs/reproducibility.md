# Reproducibility

Π-PLS separates software reproducibility from model-fitting reproducibility. The repository provides
commands for validating the source tree, installed distributions, examples, and documentation; the
modeling API exposes the settings needed to repeat a fitted analysis. Repository-maintenance details
are kept in the root `CONTRIBUTING.md`.

## Software and distribution checks

A clean source checkout should support:

```bash
python -m pip install ".[dev]"
make check
make examples
make build
make dist-check
```

`make check` validates the numerical and API contracts. `make examples` runs the maintained
application workflows. `make build` creates the wheel and source distribution, while
`make dist-check` installs each built artifact in a clean environment and exercises the installed
package outside the repository checkout.

The executable real-data examples require the dedicated example dependencies when they are not
already installed through the development extra:

```bash
python -m pip install ".[examples]"
make examples
```

Randomized numerical paths and synthetic generators expose explicit random-state controls. The
package-owned reference-data loaders require no optional dataframe dependency.

## Documentation reproducibility

Install the documentation dependencies and build or preview the site from a checkout:

```bash
python -m pip install ".[docs]"
make docs-figures
make docs
make docs-serve
```

The Home comparison and maintained tutorial figures are generated from repository calculations
rather than committed image binaries. Their manifests record the inputs and selections needed to
check the generated assets, together with artifact hashes. Generated directories are ignored by
Git and removed by `make clean`.

For the stronger distribution check, run:

```bash
make docs-dist
```

This rebuilds the strict documentation site from a clean source distribution and verifies that the
inputs needed to regenerate the documented figures are included. Detailed maintainer workflows for
figure generation, packaging, and release validation are documented in `CONTRIBUTING.md`.

## Model-fitting reproducibility

`PiPLSRegression.random_state` accepts an integer seed, a NumPy `RandomState`, or `None`. The
default integer `0` is reproducible. `None` uses NumPy's global random state and should be chosen
only when repeatability is not required. The synthetic generator retains its separate explicit
integer-seed contract.

Centering and optional scaling are integral to `PiPLSRegression.fit`. The `scale` parameter remains
the default policy for both blocks, while `scale_x` and `scale_y` can override predictor and response
scaling independently. `PiPLSSearchCV` clones the complete fixed estimator or supported pipeline
inside every training fold, so each candidate learns statistics only from that fold. A subsequent
`search.refit(X, Y, ...)` call fits one selected $(h,r_\pi)$ pair on the complete supplied training
set, so centering and any enabled estimator or pipeline scaling are learned again from that full
data. Learned scaling must not be fitted globally before CV.

Record `response_subspace` as part of every model-fitting protocol. The package default
[`"cross_covariance"` response-subspace construction](theory.md#response-subspace-selection) is the
construction used in the [peer-reviewed companion publication](citation.md#companion-paper);
`"least_squares"` is a software extension outside that publication. `PiPLSSearchCV` clones this
setting unchanged and does not search over it automatically. When comparing the two policies,
materialize one validation protocol and reuse the same split indices for both searches so that
differences are not confounded with resampling variation. Record the policy together with scaling,
SVD configuration, rank-search policy, component-selection rule, and random seeds.

Maintained examples that use one ordinary five-fold regression partition specify
`KFold(n_splits=5, shuffle=True, random_state=0)` rather than relying on the package default. The
complete Pulp workflow instead uses ten seeded repetitions of five-fold CV. These choices avoid fold
assignments determined by file row order, and every stochastic maintained partition has an explicit
seed.

## Synthetic validation

`pipls.datasets` provides a deterministic latent-structure generator with known shared,
predictor-specific, and response-specific components. The
[Datasets and generators API](api/datasets.md#synthetic-generator) defines the Gaussian latent
distribution and seeded generator contract.

Scientific comparison studies, large simulation grids, paper-only comparators, and manuscript
figure or table orchestration remain downstream assets that pin a released package version. The
package test suite protects maintained numerical and API contracts directly.

## Reference datasets

[Pulp](datasets.md#pulp-real-data-integration),
[Sugarcane](datasets.md#sugarcane-spectral-integration), and
[Tobacco](datasets.md#tobacco-spectral-integration) are canonical package resources under
`src/pipls/_data/<dataset>/` and are included in built distributions. Their named loaders return the
packaged matrices without learned preprocessing.

The [dataset documentation](datasets.md) gives the original-source citation, licensing, adaptation,
dimensions, and raw-resource layout for each dataset. Package-resource tests verify the distributed
resources and canonical numeric arrays; metadata supports provenance and review but is not required
when users supply their own `X` and `Y`.

## Scope of reproducibility

The repository validates the installable package, its documented datasets, and its maintained
examples. Application-specific studies may add their own simulations, comparators, tables, and
figures without changing the package reproducibility contract.
