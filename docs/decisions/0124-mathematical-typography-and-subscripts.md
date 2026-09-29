# Decision 0124: mathematical typography and descriptive subscripts

## Status

Accepted and implemented.

## Context

The public theory pages already distinguished complete matrices from indexed columns and elements,
but the same convention had not been applied consistently to synthetic-data documentation and
maintainer contracts. Generated dataset docstrings also used reStructuredText ``.. math::`` blocks,
which were not processed reliably by the Markdown/Arithmatex documentation pipeline. Descriptive
subscripts such as the predictor, shared, response, block, and extremum labels remained italic and
could be mistaken for variable indices.

## Decision

Use one mathematical typography convention across living public documentation, generated public
docstrings, and normative maintainer contracts:

- complete matrices are bold;
- Latin matrix symbols use `\mathbf`, while Greek matrix symbols that must render in bold use
  `\boldsymbol`;
- descriptive role, block, method, and extremum subscripts use upright `\mathrm`, for example
  $d_{\mathrm{p}}$, $\boldsymbol{\Lambda}_{\mathrm{s}}$,
  $\mathbf{L}_{\mathrm{sp}}$, $\mathbf{U}_{\mathrm{X}}$, and $h_{\mathrm{max}}$;
- mathematical indices and dimensions remain italic, for example $D_k$, $D_{kk}$, $P_{:k}$,
  $s_i$, $r_\pi$, and $\mathbf{I}_p$;
- display mathematics in public Markdown and generated Markdown docstrings uses the supported
  equation environment rather than reStructuredText math directives.

Apply the convention to the public synthetic-generator model, the public Pi-PLS theory,
path-selection extrema, generated dataset API docstrings, and the corresponding normative
guide-layer equations. Historical decision records are not rewritten solely for typographic
modernization.

## Consequences

- The dataset API renders the synthetic generator's defining equation through the same Markdown
  mathematics pipeline as the public theory page.
- Predictor/shared/response labels are visibly distinct from variable indices.
- Complete matrices, indexed columns, scalar elements, and dimensional indices retain separate
  visual roles.
- No estimator, generator, search, numerical, example, dataset, or public API behavior changes.
