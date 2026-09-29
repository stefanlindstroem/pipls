# Troubleshooting

This page covers common public-API failures and unexpected behavior. Detailed estimator and
statistical contracts remain on their canonical reference pages.

## I need to change scaling, response-subspace selection, or the SVD solver during search

Configure those settings on the `PiPLSRegression` template passed as `estimator=` to
`PiPLSSearchCV`. Search changes only `n_components` and `predictor_rank`; the remaining template
configuration is cloned into every candidate. With `estimator=None`, the ordinary
`PiPLSRegression` defaults apply.

The search does not compare response-subspace policies automatically. Use
`response_subspace="cross_covariance"` for the formulation in the peer-reviewed companion
publication; if you need the
`"least_squares"` software extension, set it on the estimator template and keep it fixed across the
search. See [Pipelines and fold-local preprocessing](path_selection.md#pipelines-and-fold-local-preprocessing).

## A component count or predictor rank was not evaluated

Inspect `search.component_path_.n_components` and, for one component count,
`search.predictor_rank_profile(h).predictor_rank`. Candidate availability depends on fold-local
sample counts and numerical ranks, configured bounds, and the predictor-rank policy. An explicit
component request must have at least one admissible predictor rank. See
[Search domain](path_selection.md#search-bounds).

## The path search is too slow or uses too much memory

The dominant work is approximately the number of evaluated candidate pairs times the number of
materialized validation splits. Inspect `search.cv_results_["n_components"].size` and
`search.n_splits_` before changing the protocol.

Restricting candidates or repetitions changes the statistical evidence; randomized SVD changes the
numerical route. `n_jobs` changes execution only, but can increase memory use. Reuse one OOF report
instead of recomputing it for each downstream output. See
[Computational consequences](path_selection.md#computational-consequences).

## The path search has no admissible candidate

Reduce the requested component count or predictor-rank range, provide more observations, remove
redundant predictors, or review an explicit `max_predictor_rank`. With
`predictor_rank_values="epv"`, also review `samples_per_predictor_rank`. Specialized splitters can
reduce the smallest training-fold size or verified fold rank. See
[Cross-validation protocols and metadata](path_selection.md#cross-validation-protocols-and-metadata).

## `predict()` is unavailable after path selection

`PiPLSSearchCV` evaluates a path; it is not the final prediction model. Use `search.refit(...)` with
a component count, a named selection rule, or an existing compatible `PiPLSSelection`, then call
`predict()` on the returned `PiPLSRegression`. The search does not retain that final model. See the
[`PiPLSSearchCV` reference](api/path.md).

## The CV-MSE minimum, best candidate, and selected model disagree

They need not coincide. `best_score` follows the configured score on the predictor-rank-conditioned
component path. `minimum_cv_mse` applies its component-count tolerance rule to the component-path
CV-MSE values. Predictor-rank tolerance can already have retained a smaller rank than the exact
conditional optimum, and a custom scorer need not have the same optimum as CV-MSE.

Use `search.predictor_rank_profile(h)` for rank-level evidence and pass the same selection to
`oof_report()` and `refit()` when both operations must describe one exact row. See
[Adaptive search and scoring](path_selection.md#scoring-and-conditioned-path-selection).

## A grouped splitter reports missing metadata

Pass `groups=` to `PiPLSSearchCV.fit(X, y, groups=...)`. Do not pass split metadata to
`PiPLSRegression.fit()`, which performs no cross-validation. See
[Cross-validation protocols and metadata](path_selection.md#cross-validation-protocols-and-metadata).

## Fitting rejects the data or leaves the estimator unfitted

Public fits reject nonfinite inputs, constant columns that make the requested model undefined,
infeasible ranks, and numerical results that cannot be represented as finite `float64`. A failed fit
is transactional: partial fitted state and any earlier fitted model are removed. Correct the data or
parameters and call `fit()` again.

## Inspection rejects an extreme derived quantity

Inspection helpers reject a derived norm, squared residual, standardized value, or factor product
that cannot be represented as finite `float64`. Rescale the physical units or inspect the preceding
fitted and prediction arrays; inspection records are never returned with nonfinite numerical
contents.

## `PredictorRankSupportWarning` appears

A direct fixed fit warns when the observation count is small relative to the requested predictor
rank. The warning does not change the model. Use a lower predictor rank, provide more observations,
or use `PiPLSSearchCV` when rank selection is required. Algebraically or numerically infeasible
ranks remain errors. See
[`PredictorRankSupportWarning`](api/regression.md#pipls.PredictorRankSupportWarning).

## `copy=False` changed an input array

`copy=False` permits in-place centering and scaling of independent writable arrays. Use the default
`copy=True` when input preservation matters. Read-only arrays and overlapping predictor/response
storage are copied internally when mutation would be unsafe.

## An example cannot import Matplotlib

Plotting dependencies are optional and are not required by the estimators or inspection API. Install
the example extra before running maintained plotting examples:

```bash
python -m pip install ".[examples]"
```

Examples render immutable inspection results with Matplotlib; optional label-placement behavior does
not change the numerical results.
