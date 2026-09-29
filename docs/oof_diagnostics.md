# OOF diagnostics

This page defines out-of-fold reporting for one already accepted `PiPLSSelection`. In the
documented workflow, the component path and, when useful, the conditional predictor-rank profile
complete model selection before `oof_report()` is called. The report diagnoses that fixed
selection; it does not select another model, alter search state, or fit the final full-data model.

For the search evidence and rules that create the selection, see
[Path and selection](path_selection.md).

## Selection and split provenance { #selection-and-split-provenance }

`search.oof_report(X, y, selection=...)` validates an existing selection against the fitted search,
fits that fixed pair once per stored training fold, and returns an immutable `PiPLSOOFReport`.

The search materializes and stores defensive read-only copies of the exact validation indices used
by `fit()`. The report describes
the same validation protocol that produced the supplied selection.

## Ordered out-of-fold predictions { #ordered-out-of-fold-predictions }

The report preserves path entry order. Repeated validation predictions are averaged and their counts
are retained; uncovered entries have count 0 and NaN predictions. `has_complete_oof_coverage` records
whether every entry received at least one validation prediction, and `pooled_oof_r2` uses only entries
with OOF coverage.

The supplied data must have the same sample count, predictor count, and response-column count as the
fitted search. The search does not retain and compare the original values, so the caller is
responsible for passing the same observations in the same entry order.

## Prediction diagnostics and interpretation { #oof-prediction-diagnostics }

`pooled_oof_r2` is calculated over path entries with OOF coverage. It is a pooled statistic over the
selection-conditioned predictions, not mean foldwise $R^2$. Response-wise $R^2$, standardized RMSE,
and residual diagnostics can be obtained by passing the covered observed and OOF-predicted responses
to [`prediction_diagnostics()`](model_inspection.md#prediction-diagnostics) with
`prediction_kind="selection-conditioned OOF predictions"`.

Same-search OOF diagnostics are selection-conditioned because the development data and search
protocol already produced the supplied selection. They are therefore not an unbiased estimate of
post-selection performance. If OOF diagnostics are used to compare and retune alternative
selections, they become additional model-selection evidence; subsequent performance claims require
an appropriate outer assessment such as nested cross-validation or untouched external test data.

## OOF computation { #oof-computation }

Each `oof_report()` call fits the selected $(h,r_\pi)$ pair once per stored validation split. With
repeated
validation, repeated predictions for the same observation are averaged after those split-specific
fits and the number of contributing predictions is retained. Reusing one immutable report avoids
repeating this selected-pair fitting for each downstream diagnostic. `n_jobs` changes execution only;
OOF reporting parallelizes the selected-pair fits across validation splits.

## Result object

`PiPLSOOFReport` stores the exact selection, ordered OOF predictions, per-row prediction counts, and
pooled OOF $R^2$. Its fields are documented below from the public NumPy-style docstring.

::: pipls.validation.PiPLSOOFReport
    options:
      show_signature: false
      members:
        - has_complete_oof_coverage
