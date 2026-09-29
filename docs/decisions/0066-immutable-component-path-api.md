# Decision 0066: immutable component-path API

## Status

Accepted and implemented. Decisions 0143, 0146, and 0148 define search-owned selection,
component-count tolerance provenance, and conditional predictor-rank evidence.

## Context

A fitted search needs a concise component-path result with one conditionally selected predictor
rank for every evaluated component count. The result must remain stable after fitting, safe to
serialize, and independent of the large candidate-level `cv_results_` mapping.

Selection is a separate concern. A path is aligned numerical evidence; the fitted search owns named
selection rules, manual component-count lookup, full-data refitting, and OOF reporting.

## Decision

`PiPLSSearchCV.component_path_` is an immutable `PiPLSComponentPath` containing aligned read-only
arrays:

```text
n_components
predictor_rank
mean_test_score
cv_mse_mean
cv_mse_std
```

and path-wide scalar provenance:

```text
predictor_rank_policy
n_splits
```

Decision 0148 adds an immutable `predictor_rank_evidence` sequence aligned with optimized path rows;
fixed and EPV policies use `None`. Each row contains the predictor rank selected conditionally
for that component count under the accepted configured-score tolerance rule. Component counts are
unique and strictly increasing. `cv_mse_std` is population SD across the materialized validation
splits.

The path exposes no public selected-row lookup or recommendation methods. Use:

```python
selection = search.select(rule="minimum_cv_mse")
selection = search.select(rule="best_score")
selection = search.select(n_components=3)
```

Each operation returns an immutable `PiPLSSelection`. A selection contains one evaluated rank pair,
its score and CV-MSE summaries, the path policy, split count, and optional rule provenance. A
minimum-CV-MSE selection additionally retains the exact unruled minimum path row, resolved
component-count relative and absolute tolerances, and a derived effective threshold. Decision 0148
adds independent predictor-rank evidence to optimized selections and their reference rows.

Public path/profile arrays are defensive read-only copies and recover that storage contract after
unpickling. Semantic validity belongs to the fitted search that produces these returned-first
records; arbitrary malformed direct construction is not a supported validation workflow.

`cv_results_` remains the candidate-level evidence table. It is not duplicated inside the concise
path or scalar selections.

## Consequences

- Component-path evidence has one stable row per component count.
- Selection ownership is unambiguous and search-owned.
- No `best_*` fitted state, path recommendation method, compatibility alias, or mutable result view
  is required.
- User code can serialize, compare, and plot path arrays without retaining training data.
