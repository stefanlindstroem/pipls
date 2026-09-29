# Decision 0045: separate comparison models from shared PLS-family analysis

## Status

Accepted and implemented.

## Context

Ordinary scikit-learn `PLSRegression` is useful as an established reference curve for component-path
CV-MSE. It should not, however, become a second fitted model for the OOF predictions, latent
structure, coefficients, biplots, or observation diagnostics in a Pi-PLS analysis.

Scores, reconstruction loadings, coefficients, balanced score-loading biplots, and observation or
prediction diagnostics are shared PLS-family techniques. Pi-PLS also exposes the method-specific
factorization

\begin{equation}
B_{\mathrm{cs}}=PDQ^\mathsf{T},
\end{equation}

which requires explicitly Pi-PLS-specific inspection.

## Decision

The repository distinguishes three analysis roles.

### Comparative component-path diagnostics

`examples/03_pls_path_comparison.py` is the sole maintained PLS-family component-path comparison.
For Pulp, Sugarcane, Tobacco, and one fixed deterministic synthetic stress case, it evaluates the
publication-default and least-squares Pi-PLS response-subspace policies alongside ordinary PLS on
exactly the same materialized validation partitions and response-standardized CV-MSE summary. The
two Pi-PLS policies remain fixed-estimator alternatives rather than an additional search dimension.
Ordinary `PLSRegression` is permitted in that focused comparison because it answers a
model-comparison question; it does not create the model interpreted by the subsequent dataset
analyses. Comparative CV-MSE paths are model-development evidence rather than independent
post-selection validation.

### Pi-PLS-specific factorization inspection

Inspection of $P$, $D$, $Q$, $QD$, and the identity $PDQ^\mathsf{T}$ is specific to Pi-PLS. The
numerical API therefore retains the explicit name `pipls_display_factors()`. Rendering is
caller-owned under Decision 0083.

The terminology is predictor rotations or predictor directions for $P$, dilation values for $D$,
and response rotations or dilation-weighted response directions for $Q$ and $QD$. These quantities
must not be presented as ordinary PLS loadings.

### Shared PLS-family analysis

The following analyses use estimator-neutral numerical terminology:

- X-score and loading inspection;
- Y-loading inspection;
- regression-coefficient inspection;
- balanced score-loading biplots;
- score-distance and X-reconstruction-residual diagnostics;
- observed-versus-predicted and residual diagnostics.

Their public numerical inspection functions may accept a fitted `PLSRegression` or
`PiPLSRegression` when the object exposes the required fitted attributes and transformations. The
API validates that structural contract rather than restricting use through a concrete estimator
type. Rendering remains caller-owned.

The maintained Pulp, Sugarcane, and Tobacco analyses apply these shared tools to one selected
`PiPLSRegression`. They do not fit an ordinary PLS model for OOF prediction or post-selection
interpretation. Ordinary PLS remains confined to the explicit component-path comparison.

## Consequences

- Users retain a focused Pi-PLS-versus-PLS CV-MSE comparison.
- Every maintained post-selection report interprets and diagnoses one selected Pi-PLS model.
- Pi-PLS-specific and shared PLS-family concepts are distinguished by ownership and naming.
- Shared numerical inspection remains usable with compatible ordinary PLS models.
- The estimator, numerical core, path search, CV-MSE definitions, and Pi-PLS factorization
  identities are unchanged by this boundary.
