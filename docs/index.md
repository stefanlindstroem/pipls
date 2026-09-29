# Π-PLS documentation

`pipls` is a Python package for the multivariate linear regression model panoramic partial least squares (Π-PLS), where Π stands for the Greek πανοραμικός (“panoramic”).
For routine modeling, Π-PLS can be used much like ordinary PLS: choose a component count, fit
and predict, inspect latent scores and loadings, examine regression coefficients, and assess
observed-versus-predicted values and residuals. The package exposes those familiar PLS-family
analysis quantities together with the additional Π-PLS-specific paired-direction
factorization.

## Why use Π-PLS?

In the problems examined in the [companion paper](citation.md#companion-paper), Π-PLS is
reported to be comparably robust to ordinary PLS while matching or improving its predictive
performance; the predictive improvement is substantial in some settings. These are empirical
results which may not hold for every dataset.

**Fewer shared components can be sufficient.** For many datasets, including the
[Pulp](datasets.md#pulp-real-data-integration) and
[Tobacco](datasets.md#tobacco-spectral-integration) datasets available through this package, Π-PLS
reaches a low cross-validated prediction-error region with fewer shared components than ordinary
PLS, as seen in the following figure:

<div class="grid" markdown>
![Pulp component-path comparison between Π-PLS and ordinary PLS](assets/generated/home/pulp_component_parsimony.svg)

![Tobacco component-path comparison between Π-PLS and ordinary PLS](assets/generated/home/tobacco_component_parsimony.svg)
</div>

Both panels use the same seeded five-fold validation splits for Π-PLS and ordinary PLS, and the error bars show the split-to-split standard deviation. 

**Π-PLS gives a one-to-one relation between predictor and response directions.** The Π-PLS method differs from ordinary PLS in how it constructs the latent regression map. Π-PLS
represents that map through [paired latent modes](theory.md#diagonal-latent-coupling). Each retained
mode contains one orthonormal predictor direction, one orthonormal response direction, and one
nonnegative dilation. This structure adds method-specific interpretation without replacing the
standard PLS-family analysis workflow.

## Installation

From a source checkout, install the runtime package in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
```

Install the plotting dependencies used by the examples and tutorials with
`python -m pip install ".[examples]"`. Contributor setup is documented in the repository root
`CONTRIBUTING.md`; supported Python and dependency ranges are listed in the
[compatibility policy](compatibility.md).

## Leakage-safe modeling and validation

Cross-validation fits learned preprocessing inside each training fold. Candidate selection,
selection-conditioned out-of-fold diagnostics, final full-data refitting, and independent
post-selection assessment are kept distinct: use nested cross-validation or an independent test set
when an independent performance estimate is required. See [Path and selection](path_selection.md)
and [OOF diagnostics](oof_diagnostics.md) for the complete contracts.

## Quick start with Pulp dataset

The installed package contains the multivariate
[Pulp dataset](datasets.md#pulp-real-data-integration), for which a Π-PLS multivariate regression
model is created in one line of code:

```python
from pipls import PiPLSSearchCV
from pipls.datasets import load_pulp

X, Y = load_pulp(return_X_y=True)
model = PiPLSSearchCV().fit(X, Y).refit(X, Y, rule="minimum_cv_mse")
Y_fitted = model.predict(X)
```

This is a compact calibration-fit demonstration: `Y_fitted` comes from
the same observations used to fit the final model. Study the complete [quick start tutorial](tutorials/quick_start.md) with prediction diagnostics.

## Choose a tutorial

After the quick start, [Inspect and select with synthetic data](tutorials/synthetic.md) introduces
the complete selection contract in a small deterministic problem: inspect the unselected component
path, choose a paired-mode count and create one selection, inspect the selected path and conditional
predictor rank, refit that exact row, and predict an independent test set.

Continue with [Pulp: a complete Π-PLS workflow](tutorials/pulp.md) for real-data loading,
selection-conditioned out-of-fold (OOF) inspection before final refitting, immutable inspection results, and
representative interpretation of standard PLS-family plots and Π-PLS-specific plots.

