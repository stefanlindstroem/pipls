# Decision 0024: package-product repository boundary

## Status

Accepted and implemented.

## Context

Pi-PLS is expected to evolve beyond one publication. Organizing the package repository around one
manuscript would blur the public software contract and burden programming users with publication
figures, complete comparison grids, cached results, and orchestration that are not part of ordinary
package use.

The package must remain useful to future publications without becoming a reproduction repository
for any one of them.

## Decision

- Treat `pipls` as the long-lived repository for the installable package, public API, user
  documentation, concise examples, transparent package-owned datasets, tests, packaging, and
  releases.
- Keep manuscript figures, complete publication experiments, paper-only comparison grids, cached
  paper results, and publication-specific environments in separate downstream reproduction
  repositories.
- Let downstream reproduction repositories depend on tagged `pipls` releases.
- Permit ordinary PLS comparisons in package examples when they clarify user-relevant Pi-PLS
  behavior.
- Include OLS or CCA only when they test a package-level mathematical identity, limiting case, or
  public contract; do not add them merely to reproduce a paper comparison.
- Keep software validation in tests and focused examples rather than maintaining a separate
  benchmark product, result schema, or benchmark-output layer.
- Preserve preprocessing, alternative standardization, and block scaling as possible future product
  directions, but create no API, naming, schedule, or implementation commitment until a dedicated
  future design is accepted.

## Consequences

- The package roadmap is product-oriented and can support multiple future publications.
- Publication reproduction remains possible without becoming part of the supported package
  surface.
- Package examples and tests focus on software behavior and user-relevant comparisons rather than
  publication claims.
- No preprocessing, alternative-standardization, or block-scaling API commitment is created by this
  decision.
