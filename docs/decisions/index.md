# Decision records

This index lists current numbered decisions only. Completed intermediate records are summarized in
[history.md](history.md), and every removed filename is mapped in
[retirements.md](retirements.md). Git remains the complete archive.

## Current decisions

- [0001: fixed Pi-PLS construction](0001-core-definition.md)
- [0002: centering and scaling](0002-preprocessing-semantics.md)
- [0004: selection loss](0004-response-standardized-mse.md)
- [0007: exhaustive versus adaptive search](0007-predictor-rank-search-policies.md)
- [0008: scalable predictor decomposition](0008-predictor-svd-policy.md)
- [0009: exposed controls](0009-public-parameter-validation.md)
- [0014: groups and weighting boundary](0014-validation-metadata-scope.md)
- [0015: dataset and synthetic boundary](0015-dataset-and-synthetic-api.md)
- [0024: package versus publication ownership](0024-package-product-repository-boundary.md)
- [0025: current versus future scaling](0025-model-internal-standardization-boundary.md)
- [0039: estimator versus selection ownership](0039-fixed-estimator-path-search-boundary.md)
- [0041: legacy dataset licensing and roadmap](0041-legacy-dataset-licensing-roadmap.md)
- [0042: fitted-model analysis architecture](0042-model-inspection-and-post-analysis.md)
- [0045: comparison versus fitted-model analysis ownership](0045-pls-family-analysis-boundary.md)
- [0054: supported interpreter and dependency ranges](0054-compatibility-policy.md)
- [0066: concise path-result API](0066-immutable-component-path-api.md)
- [0072: conditional predictor-rank inspection](0072-conditional-predictor-rank-profile.md)
- [0083: final data-first rendering policy](0083-data-first-rendering-policy.md)
- [0091: clean committed-tree snapshots](0091-clean-git-snapshots.md)
- [0092: fold numerical-rank feasibility](0092-fold-numerical-rank-feasibility.md)
- [0093: immutable core public-result invariants](0093-public-result-invariants.md)
- [0094: immutable and finite inspection results](0094-inspection-result-safety.md)
- [0103: optional dependencies](0103-installation-and-optional-dependency-boundary.md)
- [0110: response-anchored Pi-PLS display factors](0110-response-anchored-display-factors.md)
- [0117: license, authorship, and citation](0117-commercial-license-authorship-and-citation.md)
- [0119: synthetic-data generator](0119-manuscript-latent-geometry-generator.md)
- [0120: companion-manuscript theory alignment](0120-companion-manuscript-theory-alignment.md)
- [0121: canonical Pi-PLS terminology](0121-canonical-pipls-terminology.md)
- [0124: mathematical typography](0124-mathematical-typography-and-subscripts.md)
- [0142: package-owned reference datasets](0142-package-owned-reference-datasets.md)
- [0143: selection provenance and OOF](0143-model-selection-provenance-and-oof-reporting.md)
- [0146: CV-MSE tolerance selection and split-SD reporting](0146-cv-mse-tolerance-selection.md)
- [0147: decision lifecycle](0147-decision-lifecycle-and-maintainer-context.md)
- [0148: predictor-rank tolerance selection](0148-predictor-rank-tolerance-selection.md)
- [0153: independent predictor and response scaling controls](0153-independent-block-scaling-controls.md)
- [0154: full-domain predictor-rank selection and explicit EPV policy](0154-full-domain-predictor-rank-selection.md)
- [0155: response-subspace selection policies](0155-response-subspace-selection-policies.md)
- [0164: lean reference architecture](0164-lean-reference-architecture.md)
- [0165: selection evidence and OOF diagnostic boundary](0165-selection-evidence-and-oof-diagnostic-boundary.md)

## Historical navigation

- [Compact development history](history.md)
- [Explicit retirement map](retirements.md)

New decision numbers are never reused. The two inherited 0153/0154 collisions are frozen and
documented in the [retirement map](retirements.md#frozen-legacy-number-collisions). Add or retire
records only through the lifecycle defined by Decision 0147.
