# Decision 0143: model-selection provenance and OOF reporting

## Status

Accepted and implemented. Decisions 0146 and 0148 refine component-count and conditional
predictor-rank provenance. This decision is the canonical contract for exact selection handoff,
generic OOF reporting, and protocol-neutral report contents.

## Context

Path evaluation, final fitting, and out-of-fold diagnostics are separate operations. Repeating a
selection request independently across those operations can silently fit one parameterization and
report another. A refitted model therefore needs to retain the exact immutable path row that
configured it, and OOF reporting needs to consume that existing selection rather than resolve a
second rule or component count.

The search already owns the materialized validation splits. Reusing those exact splits preserves
the fitted search protocol, including repeated cross-validation and custom splitters, without
creating a second splitter or an independent performance estimate.

## Decision

### Search fitting produces evidence only

`PiPLSSearchCV.fit(X, y)` materializes one validation split set, evaluates the admissible candidate
path, and retains search evidence. It retains neither training matrices nor a final selected
full-data estimator. Search evidence includes `cv_results_`, `component_path_`, `n_splits_`,
`max_predictor_rank_`, `search_is_exhaustive_`, and `scorer_`.

`PiPLSSearchCV.select()` is the sole public selected-row lookup. Exactly one of `rule` and
`n_components` is supplied, the method performs no fitting or search mutation, and it returns an
immutable `PiPLSSelection`. The named rules are:

- `best_score`: the maximum configured-score row on the predictor-rank-conditioned component path,
  with deterministic smaller-component and smaller-rank ordering for exact numerical ties;
- `minimum_cv_mse`: the smallest conditioned row satisfying the relative and absolute CV-MSE
  tolerances defined by Decision 0146.

Manual component-count selection returns the stored conditioned row for that evaluated count with
no named-rule provenance. `PiPLSComponentPath` remains aligned numerical evidence and exposes no
public recommendation or selected-row lookup methods; conditional predictor-rank inspection remains
a search operation through `predictor_rank_profile()`.

### Refitted models retain the exact selection

Every estimator or terminal pipeline returned successfully by `PiPLSSearchCV.refit()` exposes:

```python
model.selection_
```

The value is the exact immutable `PiPLSSelection` used to configure the full-data fit. It is
attached to the returned outer object only after fitting succeeds. A `PiPLSRegression` fitted
directly has no `selection_` attribute because no search-owned selection occurred.

`refit()` returns only the fitted estimator. It can resolve a rule or component count directly or
consume an existing exactly compatible `PiPLSSelection`. In the latter case the exact supplied
immutable object becomes `model.selection_`. The method does not retain a selected fitted model on
the search object and does not return a wrapper or tuple.

### Selection results retain policy provenance

A `PiPLSSelection` records the selected component count, aligned predictor rank, configured score,
CV-MSE summaries, split count, and selection rule. Manual and best-score selections have no
reference minimum or tolerance provenance.

For tolerance-based `minimum_cv_mse` selection, the result also records:

```python
selection.reference_minimum
selection.relative_tolerance
selection.absolute_tolerance
selection.cv_mse_threshold
```

`reference_minimum` is the exact unruled minimum-CV-MSE path row. The threshold is derived from the
stored minimum and resolved tolerances rather than stored independently. Decision 0146 defines the
selection inequality and validation rules.

### OOF reporting consumes an existing selection

The public operation is:

```python
def oof_report(
    self,
    X: ArrayLike,
    y: ArrayLike,
    *,
    selection: PiPLSSelection,
) -> PiPLSOOFReport:
    ...
```

`oof_report()` accepts neither a rule nor a component count. It validates that `selection` is
exactly compatible with the fitted search, then refits that selected pair on defensive copies of
every validation split materialized by `fit()`.

The immutable report exposes:

```python
report.selection
report.oof_predictions
report.oof_prediction_counts
report.pooled_oof_r2
report.has_complete_oof_coverage
```

Repeated validation predictions for one observation are averaged. Their multiplicity is retained
in `oof_prediction_counts`. Uncovered rows retain `NaN` predictions and zero counts, and pooled OOF
$R^2$ uses covered rows only. Response-wise diagnostics derived from an OOF report operate on these
row-ordered, per-observation predictions after repeated held-out predictions have been combined;
they are not averages of fold-wise response diagnostics. The operation does not rescore candidates,
fit a full-data model, retain supplied matrices, or mutate the search.

An OOF report based on a search-owned selection is a selection-conditioned diagnostic. It is not
nested cross-validation or an external-test estimate. Because the same search-owned splits both
inform the retained selection and generate the report, maintained analysis may use the report for
inspection or review but must not present it as independent post-selection qualification or
validation. Independent post-selection assessment requires an outer resampling design or untouched
external data. The report carries no classifier for the validation protocol represented by the
stored splits, and the package provides no dedicated leave-one-out mode or provenance field. Users
may still supply any compatible splitter or explicit split iterable intentionally.

### Workflow order

The standard evidence-retaining model-producing workflow is:

```python
search = PiPLSSearchCV(cv=cv).fit(X, Y)
path = search.component_path_
selection = search.select(rule="minimum_cv_mse")
rank_profile = search.predictor_rank_profile(selection.n_components)
report = search.oof_report(X, Y, selection=selection)
model = search.refit(X, Y, selection=selection)
```

Manual component-count selection uses the same order with `search.select(n_components=...)`. OOF
reporting remains optional because it performs one additional fit per stored validation split. The
compact automatic route may still call `refit(..., rule=...)` without retaining a selection first. A
workflow that only inspects selection or OOF evidence, or compares paths, need not fit a final
model.

### Method roles

```text
search.select()
    inspect one immutable selection without fitting

search.refit(selection=...)
    fit one full-data model from an existing compatible selection and retain model.selection_

search.refit(rule=... or n_components=...)
    resolve and fit one full-data model directly for compact workflows

search.oof_report(selection=...)
    compute OOF diagnostics for an existing compatible selection

search.component_path_
    expose aligned component-count evidence

search.predictor_rank_profile(h)
    expose conditional predictor-rank evidence at one component count
```

## Consequences

- A fitted model records exactly which component-count and predictor-rank pair constructed it.
- OOF reporting and final refitting can consume the same pre-existing selection, preventing an
  independent second resolution request.
- Repeated-CV reports inherit every search repetition and expose the resulting prediction counts.
- Search remains the owner of path evidence and materialized splits, but not of a refitted model.
- Selection, OOF analysis, fitted-model inspection, and rendering remain separate operations.
- The removed pre-release validation-report and selected-search-state names have no aliases.
