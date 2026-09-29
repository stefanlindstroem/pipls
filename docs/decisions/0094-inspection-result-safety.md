# Decision 0094: finite inspection calculations and immutable result storage

## Status

Accepted. Revised to separate producer-side numerical validation from result-record storage.

## Context

The inspection layer returns frozen records including `LatentStructure`, `BiplotCoordinates`,
`PiPLSDisplayFactors`, `ObservationDiagnostics`, and `PredictionDiagnostics`. Inspection helpers
perform sums of squares, norms, means, covariance products, factor products, and response
standardization that can overflow even when their inputs are finite.

An earlier implementation also made every result constructor an exhaustive cross-field validation
boundary. That duplicated checks already performed while producing the records and mixed numerical
safety with storage semantics.

## Decision

1. Public inspection functions validate their user/model inputs and own the semantic consistency of
   the records they return.
2. Inspection result records store array fields as defensive read-only float64 or platform-integer
   copies. Their constructors do not independently revalidate all producer-owned cross-field
   relationships.
3. Array-containing records reconstruct through their constructors during unpickling so read-only
   NumPy storage is restored.
4. `PredictionDiagnostics` derives its dependent diagnostic arrays from observed and predicted
   responses. Checks required to carry out those calculations safely, such as aligned shapes,
   nonconstant observed responses, and finite representability, remain with that computation.
   The public `prediction_diagnostics()` function owns response normalization and prediction-kind
   validation.
5. Biplot norms use max-scaled Euclidean calculations, and the balancing factor is formed as a
   quotient of square roots rather than by first forming a potentially overflowing norm ratio.
6. Prediction centers, sample scales, standardized RMSE, and response-wise $R^2$ use scaled
   calculations. Residual, centering, standardization, and factor products are checked immediately
   for finite float64 representability.
7. Observation score covariance is formed after one common finite scaling of the centered training
   scores. This preserves the Moore--Penrose score distance while avoiding overflow in the
   covariance product. X-reconstruction residuals use scaled row-wise squared norms.
8. If a mathematically requested inspection quantity cannot be represented as finite float64, the
   producing helper raises `ValueError` naming that quantity. Expected floating-point warnings are
   suppressed only around the checked operation; nonfinite produced results never escape.
9. These safeguards do not add thresholds, probability limits, outlier labels, uncertainty claims,
   or rendering behavior.

## Consequences

Ordinary fitted-model and prediction results retain their values, shapes, field names, defensive
copies, and numerical safety. The package no longer maintains a second exhaustive validation model
for arbitrary direct construction of inspection result records.

Dataset metadata policy is owned separately by Decision 0015 and is not part of this
inspection-result contract.
