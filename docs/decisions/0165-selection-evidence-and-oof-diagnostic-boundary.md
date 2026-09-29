# Decision 0165: selection evidence and OOF diagnostic boundary

## Status

Accepted and complete. Patch 0165A establishes the methodological and documentation boundary,
Patch 0165B aligns the maintained tutorials with it, Patch 0165C splits the combined Reference page
into Path and selection plus OOF diagnostics, and Patch 0165D completes the terminology and
cross-reference audit. This decision refines Decisions 0143 and 0164
without changing their
public-API, numerical, strict-build, or general documentation-ownership contracts.

## Context

The maintained tutorials are intended to teach an explicit evidence-retaining model-development
workflow. The component path provides the primary evidence for choosing component count, and the
conditional predictor-rank profile may be inspected when the rank decision needs additional review.
Once those choices have produced one immutable `PiPLSSelection`, the same object can be passed to
OOF reporting and final refitting.

Decision 0143 already makes that implementation boundary explicit: `oof_report()` requires an
existing compatible selection, reuses the fitted search splits, does not resolve another rule or
component count, and does not mutate the search. It also defines same-search OOF predictions as
selection-conditioned diagnostics rather than independent post-selection validation.

The documentation drifted beyond that contract. The synthetic tutorial keeps selection centered on
the path and conditional rank profile, while the complete Pulp tutorial currently describes OOF
behavior as part of the model-selection evidence and permits it to send the user back to component
selection. The combined **Selection and validation** Reference page likewise places path evidence,
selection rules, CV mechanics, OOF reporting, and refit provenance under one heading. That grouping
obscures the intended methodological boundary and makes the documented selection procedure more
adaptive than the tutorials otherwise imply.

## Decision

### Complete the documented selection before OOF inspection

The maintained workflow uses the component path as the primary component-count selection evidence.
Conditional predictor-rank evidence from `predictor_rank_profile(h)` may be inspected when the rank
decision at a candidate component count needs review. Selection is complete when the user accepts
one immutable `PiPLSSelection` representing the retained pair $(h, r_\pi)$.

The documented workflow does not use response-wise OOF diagnostics, residual patterns, observation-
level OOF behavior, or other quantities derived from `oof_report()` to tune that ordinary
selection. OOF reporting therefore consumes a completed selection rather than forming another
stage of the standard selection rule.

This boundary does not prohibit a user from designing a different adaptive procedure. If OOF
results are compared across alternatives and the selected $(h, r_\pi)$ is changed because of those
results, the OOF diagnostics have become additional model-selection evidence. Claims about
post-selection predictive performance must then account for that adaptivity through an appropriate
outer validation design or untouched external test data.

### Treat OOF reporting as selection-conditioned diagnosis

After selection, `oof_report(X, y, selection=selection)` answers how that fixed selected model
behaves across the exact validation splits retained by the fitted search. The report may expose
response-specific weaknesses, poor aggregate predictive behavior, incomplete coverage, repeated-CV
multiplicity, or observation-level error patterns when combined with prediction diagnostics.

Those findings can show that the accepted modeling strategy is inadequate for its intended use.
The maintained workflow may therefore advise reconsidering the analysis assumptions, search design,
preprocessing, candidate domain, or validation strategy when OOF behavior is substantively
unacceptable. It does not teach casual retuning of $h$ or $r_\pi$ from the same OOF diagnostic view
while continuing to describe the original path-based procedure as the selection method.

Same-search OOF diagnostics remain selection-conditioned because the development data and search
protocol that informed the retained selection also generate the OOF predictions. They are not
nested-CV or external-test estimates. Decision 0143 remains canonical for exact split reuse, report
contents, compatibility checks, repeated-prediction averaging, coverage, provenance, and selection
handoff.

### Align the tutorials around one workflow

The tutorial route teaches the following order:

```text
fit search
    -> inspect component path
    -> choose component count and create PiPLSSelection
    -> optionally inspect the conditional predictor-rank profile
    -> accept the selection
    -> inspect selection-conditioned OOF diagnostics when useful
    -> refit that same selection
    -> inspect the fitted model or assess untouched external predictions
```

The OOF step is optional and diagnostic. It may occur before or after full-data refitting as long as
both operations consume the same accepted immutable selection and their distinct provenance remains
clear. The tutorials must not show an ordinary feedback arrow from OOF diagnostics to component or
predictor-rank tuning.

### Split the combined Reference ownership

Decision 0164 remains canonical for the lean, flat, lookup-oriented Reference, early generated API
documentation, tutorial ownership of worked workflows, synthetic-generator explanation and figure,
and strict served-documentation boundary. This decision replaces only Decision 0164's combined
**Selection and validation** page ownership and resulting seven-page count.

The target flat Reference becomes:

1. **Overview**;
2. **PiPLSRegression**;
3. **PiPLSSearchCV**;
4. **Path and selection**;
5. **OOF diagnostics**;
6. **Model inspection**;
7. **Datasets and generators**;
8. **Troubleshooting**.

**Path and selection** answers what search evidence is produced and how it determines the retained
$(h, r_\pi)$. It owns search-domain and feasibility semantics, predictor-rank policy, component-path
and conditional-rank evidence, scoring, CV metadata needed to interpret that evidence, selection
rules and tolerance provenance, and the immutable search-result records used to inspect or retain a
selection.

**OOF diagnostics** answers what can be learned from the stored validation predictions for an
already established selection. It owns `PiPLSOOFReport`, exact split reuse, row ordering, repeated-CV
prediction counts and averaging, incomplete coverage, pooled and response-wise OOF diagnostics,
and the selection-conditioned versus independent-assessment interpretation boundary.

Cross-validation mechanics may be described on Path and selection when they define how path evidence
is produced. OOF-specific consequences of the retained splits belong on OOF diagnostics. The pages
cross-link at the immutable `PiPLSSelection` boundary rather than repeating each other's contracts.

### Preserve implementation and validation ownership

No Python source, numerical result, public signature, default, dataset, generator, or maintained
figure changes under this decision. `PiPLSSearchCV.select()`, `oof_report()`, and `refit()` retain
their Decision 0143 behavior. Existing selection and OOF tests continue to protect executable
contracts rather than prescribed tutorial prose or page arrangement.

Strict `make docs` and `make docs-dist` builds own served-page, link, anchor, generated-API, and
asset integration. Ordinary pytest does not freeze the eight-page navigation or wording of the
selection/diagnostic distinction.

## Patch sequence

1. **0165A** -- establish this decision and synchronize current maintainer guidance;
2. **0165B** -- align the tutorials with path/rank-profile selection followed by optional OOF
   diagnosis;
3. **0165C** -- replace Selection and validation with Path and selection plus OOF diagnostics;
4. **0165D** -- audit current served documentation, decisions, examples, and maintainer context
   for stale terminology and cross-references, then close the migration.

## Consequences

The documented selection procedure becomes easier to state and audit: path evidence selects the
model, while OOF diagnostics characterize that accepted selection. The package remains explicit
about the limitation that those diagnostics are selection-conditioned rather than independent
performance evidence.

The Reference gains one page but becomes more coherent. Users looking up selection rules do not
need to work through OOF-report semantics, and users interpreting OOF results receive a dedicated
place for provenance and coverage contracts. The tutorial sequence and Reference then describe the
same methodological boundary without changing implemented behavior.
