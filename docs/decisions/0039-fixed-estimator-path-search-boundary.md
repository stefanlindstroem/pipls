# Decision 0039: fixed estimator and path-search boundary

## Status

Accepted and implemented. Decision 0154 owns the current full-feasible predictor-rank domain and
coverage defaults, Decision 0143 owns post-search selection/OOF/refit semantics, and Decision 0155
owns the fixed-estimator response-subspace policy.

## Context

`PiPLSRegression` once mixed fixed-rank fitting with internal predictor-rank selection while a
separate search object evaluated the triangular component/rank path. That duplicated
model-selection responsibility and made an apparently ordinary estimator fit perform hidden
cross-validation.

The package instead needs one direct estimator for an explicit Pi-PLS model and one dedicated
meta-estimator for package-owned path evaluation and selection.

## Decision

`PiPLSRegression` fits one explicit pair $(h,r_\pi)$.

- `predictor_rank` is a positive integer, not a search mode or a rule-derived ceiling.
- `fit()` performs no cross-validation or parameter selection.
- Cross-validation controls, scoring controls, OOF reporting, and search-result state do not belong
  to the fixed estimator.
- A direct fixed fit emits `PredictorRankSupportWarning` when $n/r_\pi < 3$; algebraically
  infeasible ranks remain errors.

`PiPLSSearchCV` is the standard package interface for Pi-PLS model selection.

- It evaluates the admissible triangular surface $1 \leq h \leq r_\pi \leq r_{\pi,\max}$.
- Its ordinary optimized rank domain is bounded by hard fold feasibility plus any explicit user
  restriction; the statistical $n/c$ heuristic belongs only to the explicit EPV policy.
- It materializes the validation splits, evaluates fixed-rank estimator clones, and conditionally
  retains one predictor rank for every component count.
- `fit()` stores search evidence only. It does not fit or retain a final full-data model.
- Selected-row lookup, selection-conditioned OOF reporting, and explicit full-data refitting are
  post-fit search operations governed by Decision 0143.
- Controlled internal candidate fits suppress only the expected `PredictorRankSupportWarning`; other
  warnings and errors remain visible.

Fixed `PiPLSRegression` instances remain compatible with ordinary scikit-learn meta-estimators when
users supply explicit valid rank pairs. Repository examples use `PiPLSSearchCV` for the standard
Pi-PLS path rather than presenting a rectangular external `GridSearchCV` grid as the primary
workflow.

`response_subspace` remains fixed-estimator configuration under Decision 0155. Search preserves the
configured estimator-template policy while optimizing only component count and predictor rank; a
comparison of response-subspace policies therefore uses separate estimator templates rather than a
third search dimension.

## Consequences

- Fixed fitting and model selection have separate public owners.
- Search fitting produces inspectable path evidence but no hidden selected estimator.
- Ordinary path analysis uses one CV layer; unbiased post-selection assessment requires a separate
  outer resampling design or untouched external data.
- Decision 0154 defines both the full-sample EPV convention and the default full-feasible search
  domain with its exhaustive/adaptive coverage boundary.
- Supported pipelines require no public parameter-prefix control because the terminal Pi-PLS step
  is unique and inferred.
