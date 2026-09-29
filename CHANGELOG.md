# Changelog

## 0.1.0 - 2026-09-29

Initial public release of `pipls`, the Python implementation of panoramic partial least squares
(Π-PLS).

- Provide scikit-learn-style estimators for fixed Π-PLS fitting and cross-validated model
  selection across component count and retained predictor rank.
- Support prediction, selection-conditioned out-of-fold diagnostics, final refitting, and
  inspection of paired predictor and response directions, coupling strengths, and regression
  coefficients.
- Include the peer-reviewed cross-covariance construction and an explicit least-squares
  response-subspace extension.
- Ship the Pulp, Sugarcane, and Tobacco reference datasets with provenance and license notices, plus
  a validated synthetic-data generator.
- Provide maintained tutorials, executable examples, strict documentation, package-distribution
  checks, and automated tests across Python 3.10 through 3.14.
