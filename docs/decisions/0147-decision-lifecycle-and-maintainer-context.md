# Decision 0147: decision lifecycle and maintainer-context consolidation

## Status

Accepted and implemented. The original seven-patch consolidation and the bounded Patches 8--13
maintenance continuation are complete. Patches 8--11 repaired and consolidated the decision
registry and active maintainer handoff. Patch 12 removed stale implementation- and prose-policing
assertions. Patch 13 moved complete application/tutorial execution to its dedicated Make targets
and CI owners while retaining focused behavioral coverage. None of Patches 8--13 changes numerical
or public-API behavior.

## Context

At adoption, the repository contained 146 numbered decision records and more than five thousand
lines of the former active `.llm` guidance. These records document valuable architectural reasoning, but many
describe
superseded APIs, one-off migrations, completed patch sequences, or cleanup work whose durable
outcome is already captured by a later canonical decision. Keeping every intermediate record in the
active decision index makes current policy harder to identify and encourages stale wording to remain
in the maintainer context.

The same problem appeared in the former `.llm` layer. Current contracts, completed transition logs,
and historical implementation narratives are mixed together. This increases maintenance cost and
can cause a future maintainer to follow an obsolete intermediate state instead of the implemented
surface.

Git already preserves the complete repository history. The maintained tree therefore needs a clear
lifecycle for decision records and a smaller current-state guide layer, without discarding durable
scientific, numerical, API, distribution, or repository contracts.

## Decision

### Classify maintained decision material

Decision material has three maintained classes.

**Current decisions** remain as numbered records under `docs/decisions/`. A current decision still
defines at least one of:

- observable public or numerical behavior;
- a repository, distribution, documentation, or testing policy;
- a scientific or data-governance boundary;
- an unresolved accepted roadmap that still constrains implementation.

**Historical summaries** compact completed development eras into `docs/decisions/history.md`. A
summary records the durable outcome, names the current canonical decisions, and gives enough context
to understand why the repository has its present shape. It does not reproduce deleted decisions or
become a second patch log.

**Retired decisions** are removed from the maintained tree and recovered from Git history when
needed. Their decision numbers are not reused. Patch 8 records two inherited exceptions, 0153 and
0154,
where the maintained retirement map already contains older filenames with the same numeric
prefixes as current decisions. Those exact collisions are frozen historical anomalies; no new
reuse is permitted.

### Retire by relevance, not age

A numbered decision is eligible for retirement when it:

- describes a public API or repository shape that no longer exists;
- is explicitly superseded by a later canonical decision;
- records only patch ordering, migration progress, or a one-time cleanup;
- duplicates a later decision that fully defines the durable contract;
- documents an intermediate naming or compatibility state with no remaining active effect.

A decision must not be retired merely because it is old. Foundational mathematical, numerical,
scientific, data-provenance, compatibility, distribution, and repository-policy records remain
current while they still constrain the package.

### Require an explicit retirement map

Before deleting numbered records, the retirement patch must provide an explicit map from each
retired decision to one of:

- a current canonical decision;
- a section of `docs/decisions/history.md`;
- Git history alone, when the record describes only a one-off migration or cleanup.

The same patch must remove or redirect active links from the former `.llm` layer, tests, documentation, and retained
decisions. No copied archive, tarball, or hidden duplicate of the retired Markdown files is added to
the repository.

### First retirement map

Patch 3 adds `docs/decisions/retirements.md` and removes 13 records in five unambiguous groups:

- the removed Linnerud integration;
- the retired benchmark contract, runner, output schema, and focused benchmark design;
- the former Tobacco randomized-solver demonstration;
- the removed standard-error, path-recommendation, 1-SE workflow, and 1-SE figure sequence;
- the superseded constructor-selection, validation-report-composition, and selected-search-alias
  records.

Every deleted filename maps directly to one or more retained canonical decisions. Patch 4 adds the
historical summary and expands the retirement map to completed migration, naming, rendering,
documentation-arrangement, and cleanup records.

### Keep the active indexes current-focused

After consolidation:

- `docs/decisions/index.md` lists current numbered decisions and links to the historical summary;
- `docs/decisions/index.md` is the sole registry of current decisions;
- `docs/decisions/history.md` summarizes completed eras without attempting exhaustive
  reconstruction;
- Git remains the authoritative archive for retired records.

The target of approximately 35--50 current decisions is a review goal, not a test-pinned invariant.
The retained count follows the relevance criteria above.

### Compact the active maintainer layer

The maintainer contracts, now under `docs/maintainers/`, describe the implemented package and current work only. Completed patch
narratives and superseded intermediate contracts move to the historical summary or disappear when
Git history is sufficient. In particular:

- `state.md` records current implementation state;
- `strategy.md` records unresolved work and the current roadmap;
- `testing.md` records current validation obligations;
- `analysis.md` records current architectural and interpretation reasoning;
- `public_api.md` and `numerical_contracts.md` describe only the implemented surface.


### Keep validation behavior-focused

Repository tests protect executable behavior and machine-readable outputs rather than prescribed
source or documentation wording. Strict documentation, complete examples, source-distribution
documentation, and installed-artifact checks belong to their dedicated Make targets. Pytest may
parse artifacts created during a test, but it does not freeze prose, local helper names, AST call
order, plotting style, Mermaid labels, workflow files, or removed pre-release spellings.

Private maintenance helpers may consolidate subprocess, virtual-environment, archive, and artifact
mechanics under `tools/`. Large caller-owned renderers may use local helpers. Neither practice adds
installed package surface.

### Preserve behavior during documentation cleanup

Decision retirement and maintainer-contract compaction do not authorize numerical, public-API, dataset,
distribution, or rendering changes. Later source refactoring remains separately reviewable. The
public `pipls.datasets` import surface also remains unchanged when the oversized dataset module is
split internally.

## Patch sequence

1. Establish this decision and synchronize the guide-layer phase state -- complete.
2. Compact and correct the active maintainer contracts -- complete.
3. Retire explicitly superseded decisions using a reviewed retirement map -- complete.
4. Add `docs/decisions/history.md` and consolidate completed micro-decisions -- complete with 45
   current records.
5. Simplify brittle repository and workflow-structure tests and harden snapshot creation against
   tracked caches and generated artifacts -- complete.
6. Split `datasets.py` into private type, resource-loading, and synthetic-generation modules while
   preserving the public façade -- complete.
7. Normalize links and indexes, complete stale-surface and repository-artifact audits, and mark this
   decision implemented -- complete.
8. Repair the maintained registry, document the inherited 0153/0154 number collisions, and add a
   structural registry checker -- complete.
9. Retire completed records 0156--0163 after folding only durable contracts into current
   canonical records and history -- complete with 47 current records.
10. Retire older presentation/workflow records whose durable content is canonical elsewhere --
    complete with 43 current records.
11. Consolidate overlapping search-lifecycle decisions and compact the active maintainer contracts around
    current state and unresolved work -- complete with 39 current records.
12. Remove stale pytest assertions that police prose, source arrangement, private names, or removed
    pre-release spellings instead of durable behavior -- complete.
13. Move complete application/documentation validation to its owning targets and remove redundant
    ordinary-pytest execution -- complete.

## Final audit outcome

The closing audit found no broken local decision links, active references to retired decision files,
tracked cache or generated artifacts, private package imports from maintained examples or rendering
tools, obsolete pre-release API names in active user or maintainer material, unreferenced top-level
private source helpers, or uncalled example-owned plotting and rendering functions. Distribution
configuration includes the private dataset implementation modules through ordinary package
discovery while retaining the public `pipls.datasets` façade.

The active maintainer layer now records current state and unresolved work only. Complete numbered
examples run once in dedicated Python-3.12 CI through `make examples`; complete tutorial rendering
is owned by the documentation workflow through `make docs`. Ordinary pytest retains focused
numerical, API, rendering-helper, artifact, and maintenance-tool contracts without rerunning those
complete workflows. The completed consolidation sequence remains in this decision and Git history
rather than in active roadmap prose.

## Validation

Every patch must pass `git diff --check`, Python compilation, Ruff, mypy, the complete pytest suite,
and strict MkDocs validation where available.

Decision-retirement patches must additionally run `make decision-check`. The checker verifies that
every shipped numbered decision appears once in both maintained registries, local decision links
resolve, active decision references name current records, retirement-map filenames are unique, and
no decision number is reused outside the frozen 0153/0154 legacy collisions. Retirement review must
still confirm that the map covers every record deleted by the patch. Tests must not enforce a fixed
number of decisions or preserve one tombstone assertion per retired record.

The snapshot-hardening patch must demonstrate that untracked ignored caches remain excluded and
that a deliberately tracked cache or generated artifact causes snapshot creation to fail.

The dataset split must preserve public imports, packaged resource contents, deterministic synthetic
outputs, immutable result behavior, pickling, and distribution contents exactly.

## Consequences

Maintainers get a smaller current-policy surface while Git retains the complete development record.
Decision numbers remain stable and meaningful, but intermediate migration documents no longer
compete with current contracts. Future decisions enter the current registry when accepted and may
later be summarized or retired through the same explicit process.
