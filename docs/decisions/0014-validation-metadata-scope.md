# Decision 0014: validation metadata and weighting scope

## Status

Accepted after Phase D2.

## Context

Phase D2 added ordinary scikit-learn splitters, explicit group metadata, and optional OOF
predictions. Modern scikit-learn metadata routing can also propagate sample weights and arbitrary
fit/scoring metadata through meta-estimators. Pi-PLS does not require weighted model fitting for
the scientific or publication workflows currently in scope, and partial routing would create an
ambiguous public contract.

## Decision

The public fitting contract is deliberately narrow:

- `PiPLSRegression.fit` and `PiPLSSearchCV.fit` may accept keyword-only `groups` for group-aware
  splitters;
- the groups request may participate in scikit-learn metadata routing for that specific purpose;
- Pi-PLS does not implement weighted fitting, weighted predictor-rank/path selection, or
  `sample_weight` propagation through candidate fits;
- Pi-PLS does not claim general-purpose metadata routing for arbitrary nested estimators, scorers,
  or splitters;
- the optional `sample_weight` argument accepted by scalar `score` follows regressor compatibility
  conventions and does not imply weighted fitting or weighted CV selection;
- all documented response-standardized candidate losses and OOF summaries remain unweighted over
  validation rows and response columns.

Any future addition of weighted fitting or general metadata routing requires a new explicit owner
decision, mathematical/scoring definitions, public API documentation, and end-to-end tests.

## Consequences

- fresh-chat work must not infer weighted-fitting support from scikit-learn interoperability;
- D2 remains aligned with ordinary splitter and groups workflows without broadening the numerical
  method;
- public documentation must distinguish groups-only routing from general metadata routing;
- dataset and paper phases should encode sampling design through splitters and explicit data
  construction rather than sample weights.
