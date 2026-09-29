# Decision 0164: lean reference architecture

## Status

Accepted. Patch 0164A records the architecture and supersedes the former 0065 documentation-layer
record. Patches 0164B--0164F complete the original served-page migration. Decision 0165 retains
this lean-reference architecture but replaces the combined Selection and validation ownership and
the resulting seven-page count with separate Path and selection and OOF diagnostics pages.

## Context

The tutorial route now teaches the principal user workflows well enough that the Reference section
no longer needs to repeat them. Over time, however, the Reference navigation accumulated mixed
roles: estimator lookup, task guidance, statistical contracts, performance advice, and scientific
interpretation. The fixed-regression and path-selection pages became long hybrids before reaching
their generated API documentation, path-selection details and computational performance became
standalone reference chapters, and model inspection was split into nested Concepts and API pages.

A user who has already learned the package through the tutorials is more likely to enter Reference
to answer an exact programming or interpretation question: which object to use, what a parameter or
result means, how selection or validation behaves, what an inspection quantity represents, what a
dataset or generator returns, or why an operation failed. Reference should optimize for that lookup
mode while retaining exact scientific and numerical contracts that cannot be inferred from a Python
signature alone.

The existing synthetic-data documentation is a special case of useful explanatory reference
material. The three latent roles and their relationship to the observed predictor and response
blocks are necessary to understand the generator API. That explanation, including the maintained
latent-role figure, must remain served documentation even as the surrounding reference becomes
leaner.

## Decision

### Use one flat Reference section

The original target Reference navigation used seven pages. Decision 0165 later refines that target
by replacing the combined **Selection and validation** page with separate **Path and selection** and
**OOF diagnostics** pages. All other page ownership in this decision remains current.

There are no nested Reference navigation groups. In particular, Model inspection is one page rather
than separate Concepts and API children.

### Give every reference page one primary lookup question

**Overview** answers where a programming user should look. It gives a compact public-API map,
distinguishes fixed fitting from search, identifies the main result and utility areas, and links to
the canonical pages. It is a directory rather than another workflow guide.

**PiPLSRegression** answers how the fixed estimator is configured and what it returns. Generated
`PiPLSRegression` documentation appears near the start of the page. Estimator-specific manual prose
is limited to distinctions or contracts that materially aid interpretation, such as the separate
component and predictor ranks, response-subspace policy, scaling precedence, decomposition result,
and direct-fit support warning. `PiPLSDecomposition` and `PredictorRankSupportWarning` belong with
this fixed-estimator reference unless a later API change gives them a clearer owner.

**PiPLSSearchCV** answers how the search estimator is configured and what its explicit post-fit
lifecycle exposes. Generated `PiPLSSearchCV` documentation appears near the start. The page may
summarize `fit()`, `select()`, `oof_report()`, and `refit()` but does not retell the tutorial
workflow or duplicate detailed statistical selection rules.

Decision 0165 now owns the boundary between **Path and selection** and **OOF diagnostics**. The
former owns search evidence and the rules that retain one immutable selection; the latter owns the
selection-conditioned diagnostics computed for that accepted selection. The original consolidation
of durable path-selection-details and computational-accounting material remains valid, but its
served destination is refined by Decision 0165.

**Model inspection** answers what numerical quantities can be extracted from a fitted model and how
they should be interpreted. Scientific definitions and generated inspection API documentation live
on the same flat page, organized by maintained quantities or operations rather than by a
Concepts-versus-API distinction. Worked plotting procedures remain tutorial or example material.

**Datasets and generators** answers what packaged datasets, the validated dataset record, and the
synthetic generator are available and what they return. This page retains enough explanation to
understand how synthetic generation works, including the predictor-specific, shared, and
response-specific latent roles and the maintained
`docs/assets/figures/latent_geometry_generator.svg` figure. The defining equation lives here with
the generator semantics rather than being repeated on general dataset pages. Publication-specific
simulation grids, seeded studies, and result reproduction remain downstream rather than requiring
a separate served documentation page.

**Troubleshooting** answers why a supported operation failed or behaved unexpectedly. It remains a
compact task-oriented lookup page for warnings, validation failures, search/refit lifecycle errors,
numerical representability failures, copying behavior, and unexpectedly expensive operations. It
links to canonical contracts instead of reproducing them at length.

### Remove standalone reference chapters whose material has a clearer owner

The target navigation has no standalone **Path-selection details** or **Computational performance**
page. Their durable material remains consolidated into the lean Reference, with Decision 0165
assigning search/selection contracts to Path and selection and OOF-specific contracts to OOF
diagnostics. Solver behavior remains with the estimator that exposes the solver; search-domain and
parallelism cost consequences remain with `PiPLSSearchCV` or Path and selection; symptom-oriented
performance advice belongs in Troubleshooting. General optimization recommendations that merely
repeat workflow choices need not remain in Reference.

This consolidation is about ownership, not deletion by default. Material with a durable scientific,
numerical, reproducibility, dataset-provenance, or user-interpretation role must first receive a
canonical destination before its old page is removed.

### Prefer generated API documentation over duplicated manual contracts

Object-oriented reference pages place generated API documentation early. Manual Markdown provides
orientation, relationships between objects, mathematical interpretation, and package-specific
contracts that are not sufficiently clear from signatures and audited NumPy-style docstrings.
Manual prose should not restate parameter lists, field definitions, or method behavior solely to
create a second copy of the same contract.

Tutorials and examples own worked analyses and procedural teaching. Reference may contain compact
usage fragments when they clarify an object contract, but it does not reproduce complete analysis
workflows already taught elsewhere.

### Preserve the strict served-documentation boundary

The site remains a strict self-contained MkDocs build. Generated API pages derive signatures and
field documentation from audited public source docstrings. Numbered decisions and maintainer contracts remain
maintainer records excluded from served navigation and search. Documentation inputs required for a
clean build remain included in the source distribution, and checkout and source-distribution builds
continue to validate links, anchors, generated API targets, mathematics, and maintained assets.

Cross-references remain semantic rather than mechanical. Stable scientific, dataset, publication,
and API concepts should link to their canonical owner where useful; tests do not freeze incidental
prose placement or hyperlink counts.

## Consequences

Reference becomes a flat lookup surface centered on public objects and exact domain contracts rather
than a second tutorial sequence. The estimator pages can become substantially shorter without
losing behavior, while the path/selection, OOF-diagnostic, and inspection domains retain enough
manual explanation for Pi-PLS-specific semantics under the refinement in Decision 0165.

Synthetic generation remains explainable rather than collapsing into signatures alone. The
latent-role explanation, defining equation, distributional assumptions, and maintained figure stay
with Datasets and generators, while the companion-manuscript page retains only the specialized
publication-reproduction scope. The Reference datasets page owns provenance, licensing, adaptation,
and matrix facts for the packaged real datasets.

The restructuring changes documentation ownership and navigation only. It does not change numerical
behavior, estimator behavior, public API, dataset contents, generator behavior, or rendering
contracts.
