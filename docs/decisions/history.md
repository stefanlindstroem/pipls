# Decision history

This page summarizes completed development eras whose intermediate numbered decisions no longer
belong in the current policy set. It is not an exhaustive patch log and does not replace Git
history. Current behavior is defined by the retained numbered decisions, the implementation, and
the active maintainer contracts.

## Numerical and search foundation

The earliest work established the fixed Pi-PLS construction, fold-local response-standardized loss,
rank feasibility, exhaustive and adaptive predictor-rank search, and scikit-learn-compatible path
evaluation. Intermediate rank-bound rules, constructor-owned refitting, path lookup helpers, and
search-default presentation records were later replaced by the fixed-estimator/search boundary and
one explicit post-fit selection lifecycle. Materialized-split reuse and numerical low-rank ties are
covered by the search-policy/tolerance decisions; stable scorer presentation is part of the
response-standardized loss contract; and search-owned selection, OOF reporting, and refitting are
canonical in Decision 0143.

Current canonical records: [0001](0001-core-definition.md),
[0002](0002-preprocessing-semantics.md), [0004](0004-response-standardized-mse.md),
[0007](0007-predictor-rank-search-policies.md), [0008](0008-predictor-svd-policy.md),
[0009](0009-public-parameter-validation.md), [0014](0014-validation-metadata-scope.md),
[0039](0039-fixed-estimator-path-search-boundary.md),
[0066](0066-immutable-component-path-api.md), [0092](0092-fold-numerical-rank-feasibility.md),
[0143](0143-model-selection-provenance-and-oof-reporting.md),
[0146](0146-cv-mse-tolerance-selection.md), [0148](0148-predictor-rank-tolerance-selection.md),
[0154](0154-full-domain-predictor-rank-selection.md), and
[0155](0155-response-subspace-selection-policies.md).

## Datasets and product boundary

The repository moved from repository-relative example matrices and candidate integrations to a
closed, package-owned set of Pulp, Sugarcane, and Tobacco resources. Linnerud was removed, and
unlicensed legacy candidates remain excluded. Synthetic generators remain local, deterministic,
and independent of a generic registry or downloader. Publication reproduction remains downstream
of the package repository.

Current canonical records: [0015](0015-dataset-and-synthetic-api.md),
[0024](0024-package-product-repository-boundary.md),
[0025](0025-model-internal-standardization-boundary.md),
[0041](0041-legacy-dataset-licensing-roadmap.md),
[0119](0119-manuscript-latent-geometry-generator.md),
[0120](0120-companion-manuscript-theory-alignment.md), and
[0142](0142-package-owned-reference-datasets.md).

## Examples and documentation

The example series evolved through several numbering, ownership, and tutorial arrangements. The
maintained outcome is a self-contained six-example route led by an automatic Pulp fit, followed by
manual selection review, one unified PLS-family path comparison, and complete real-data
interpretation workflows. The unified comparison overlays the publication-default and least-squares
Pi-PLS response policies with ordinary PLS on matched folds for all three reference datasets plus
one fixed near-saturated synthetic stress case. The Home page derives a simplified Pulp/Tobacco
view from that same numerical protocol to illustrate possible parsimony in shared component count
without exposing the optional least-squares policy in the introductory motivation. Served
documentation is self-contained and keeps maintainer decisions outside the user site. The earlier
tutorial/guide/reference ownership split eventually allowed the Reference section to accumulate
workflow, performance, conceptual, and API material in the same navigation. Decision 0164
supersedes that arrangement with a flat lookup-oriented Reference while preserving tutorial
ownership and strict generated documentation. Semantic cross-referencing gives reference datasets,
the companion publication, and high-value mathematical concepts stable destinations without
mechanically linking every repeated occurrence.

The late comparison/navigation sequence that produced those outcomes is historical: the separate
response-subspace example was folded into Example 03; one predeclared near-saturated synthetic case
was retained despite not producing a broad policy ordering; the Home motivation was derived from
the same matched-fold comparison; and a staged cross-linking pass established useful semantic
routes. Exact case values, page layout, rollout stages, and link-placement details remain in Git
rather than current policy.

The earlier companion-manuscript guide decision and the later manual selection-review decision are
also historical implementation records. The served synthetic-data guide remains part of the user
documentation, but its durable package-versus-publication boundary is owned by Decisions 0024,
0119, and 0120. The exact tutorial diagram order and paired before/after component-path artifacts are presentation
choices. The former manual feedback arrow from OOF diagnostics to ordinary model tuning is
historical; Decision 0165 now owns the documented selection-versus-diagnosis boundary, while exact
selection handoff and same-search OOF report semantics remain owned by Decision 0143.

Current canonical records: [0164](0164-lean-reference-architecture.md),
[0165](0165-selection-evidence-and-oof-diagnostic-boundary.md),
[0143](0143-model-selection-provenance-and-oof-reporting.md),
[0155](0155-response-subspace-selection-policies.md), and the product and workflow boundaries in
[0024](0024-package-product-repository-boundary.md),
[0045](0045-pls-family-analysis-boundary.md), and
[0142](0142-package-owned-reference-datasets.md).

## Inspection and rendering

Inspection developed from package-owned plotters and report objects into validated immutable
numerical results with caller-owned rendering. Intermediate atomic plotters and direct-rendering
migration records are historical. The package owns coordinates, diagnostics, and orientation; users
and examples own Matplotlib artists, panels, labels, layout, and file output. The maintained Pulp
biplot uses one complete example-local renderer with optional line-aware `textalloc` placement and a
plain-Matplotlib fallback; tutorial readers see only the numerical biplot coordinates and one local
plotting call. This supersedes the earlier label-only and `adjustText` arrangements while preserving
the non-fatal fallback rule.

The final Pulp annotation sequence first made an external allocator optional, then replaced
`adjustText` with line-aware `textalloc`, and finally consolidated the example-local biplot into one
complete renderer with a plain-Matplotlib fallback. Exact font sizes, placement-distance tuning,
and helper-refactoring steps are historical implementation details. A later presentation migration
made response-wise selection-conditioned OOF $R^2$ the visible scalar diagnostic in the complete
real-data workflows while keeping standardized RMSE available numerically and preserving explicit
OOF versus fitted-value provenance.

The former example-owned report-composition decision and spectral predictor-rank figure decision
are historical presentation records. Direct figure/page composition is already part of the final
data-first rendering policy, while the numerical rank-profile result, split-SD meaning, and the two
sequential tolerance decisions are canonical in Decisions 0072, 0146, and 0148. Exact panel grids,
page groupings, and example-local figure arrangements remain presentation details.

Current canonical records: [0042](0042-model-inspection-and-post-analysis.md),
[0045](0045-pls-family-analysis-boundary.md),
[0072](0072-conditional-predictor-rank-profile.md),
[0083](0083-data-first-rendering-policy.md), [0094](0094-inspection-result-safety.md),
[0103](0103-installation-and-optional-dependency-boundary.md),
[0110](0110-response-anchored-display-factors.md),
[0124](0124-mathematical-typography-and-subscripts.md),
[0146](0146-cv-mse-tolerance-selection.md), and
[0148](0148-predictor-rank-tolerance-selection.md).

## Packaging and release engineering

Documentation builds, compatibility matrices, clean wheel and source-distribution tests, optional
dependency groups, generated tutorial assets, snapshot creation, and deployment were introduced in
separate increments. Their durable outcome is a strict self-contained documentation build, tested
Python and dependency ranges, isolated artifact validation, optional graphics dependencies, and
clean committed-tree snapshots.

Current canonical records: [0054](0054-compatibility-policy.md),
[0164](0164-lean-reference-architecture.md),
[0091](0091-clean-git-snapshots.md),
[0103](0103-installation-and-optional-dependency-boundary.md),
[0117](0117-commercial-license-authorship-and-citation.md), and
[0147](0147-decision-lifecycle-and-maintainer-context.md).

## Pre-release API normalization

Before the first release, repeated naming and cleanup passes removed aliases, duplicated fitted
state, shape-only properties, redundant result access, public-looking private modules, and unused
helpers. Those intermediate records are historical because no compatibility period was required at
version `0.0.0`. The current names and export boundaries are defined by the mathematical terminology,
result invariants, focused modules, and final implementation-surface decision.

Current canonical records: [0039](0039-fixed-estimator-path-search-boundary.md),
[0093](0093-public-result-invariants.md),
[0120](0120-companion-manuscript-theory-alignment.md),
[0121](0121-canonical-pipls-terminology.md), and the current API decisions listed above.

## Completed workflow, terminology, and cleanup sequences

Several late pre-release records described bounded migrations rather than independent durable
contracts. The three-stage onboarding sequence was completed and refined into the current
selection-review workflow. The search-policy rename settled on `"adaptive"` and `"exhaustive"`, the
computational-performance guidance was later consolidated into the lean Reference, exact selection
handoff was
folded into the generic provenance and OOF contract, package-owned leave-one-out support was removed,
and the behavioral test-suite cleanup replaced source and prose policing with executable boundaries.

Current canonical records: [0007](0007-predictor-rank-search-policies.md),
[0164](0164-lean-reference-architecture.md),
[0083](0083-data-first-rendering-policy.md),
[0143](0143-model-selection-provenance-and-oof-reporting.md), and
[0147](0147-decision-lifecycle-and-maintainer-context.md). The served
[Path and selection](../path_selection.md) page records the durable search, validation, selection,
and computational-accounting contracts; [OOF diagnostics](../oof_diagnostics.md) records the
selection-conditioned OOF contracts; symptom-oriented performance guidance is in
[Troubleshooting](../troubleshooting.md).

## Retired experiments and one-off cleanups

The removed benchmark layer, abandoned candidate datasets, temporary aliases, one-time navigation
changes, and patch-specific cleanup records have no remaining independent contract. Git retains the
full documents. The current retirement policy and exhaustive replacement map are defined by
[0147](0147-decision-lifecycle-and-maintainer-context.md) and
[retirements.md](retirements.md).
