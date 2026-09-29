# Decision 0121: canonical Π-PLS terminology

## Status

Accepted and implemented.

## Context

The companion manuscript gives the fixed Π-PLS construction correctly, but its terminology for
$P$ and $Q$ varies by section. Section 3.3 calls them latent basis matrices, Algorithm 1 calls them
loadings, the Corn discussion calls $P$ a projection matrix, and application prose also uses weight
factors. The manuscript writes diagonal entries as $D_k$ in places and uses “component” for several
different latent objects.

The package already makes useful distinctions. Its Π-PLS factorization exposes $P$ and $Q$
separately from `x_loadings_` and `y_loadings_`, which are least-squares reconstruction loadings.
It also exposes response-by-mode weighted directions $QD$, whereas the manuscript often presents
the transposed mode-by-response form $DQ^{\mathsf T}$. Canonical terminology is needed before
vocabulary is propagated across the broader documentation and docstring surface.

## Decision

Use the Unicode spelling `Π-PLS` for the method name in headings, prose, labels, and other
user-facing text. Reserve mathematical notation such as $\Pi$ for mathematical objects rather than
for typesetting the method name. The installable Python package remains `pipls`, and established
public Python identifiers such as `PiPLSRegression` and `PiPLSSearchCV` retain their ASCII spelling.
Bibliographic titles are quoted as published rather than rewritten.

Use the following canonical terms:

| Object or public name | Canonical term or meaning |
|---|---|
| $\Pi$ | retained predictor basis |
| $\Pi\Pi^{\mathsf T}$ | retained-subspace projector |
| $P$ | orthonormal predictor directions |
| $Q$ | orthonormal response directions |
| $D_k=D_{kk}$ | dilation of paired latent mode $k$ |
| $XP$ | predictor scores |
| $YQ$ | response scores |
| $(P_{:k},D_k,Q_{:k})$ | paired latent mode $k$ |
| `predictor_rank` | retained predictor-subspace dimension $r_\pi$ |
| `n_components` | number of paired latent modes $h$ |

“Direction” is the primary mathematical and explanatory term for columns of $P$ and $Q$. “Basis”
may describe the spans collectively. Do not call $P$ or $Q$ projection matrices: the associated
projectors are $PP^{\mathsf T}$ and $QQ^{\mathsf T}$. Do not call them reconstruction loadings;
those are the separate `x_loadings_` and `y_loadings_` quantities.

Retain the established public Python names `predictor_rotations`, `response_rotations`,
`x_rotations_`, and `y_rotations_`. This terminology decision does not rename public fields. The
word “rotation” remains acceptable when referring to those identifiers or to the orthogonal change
of latent coordinates, but it is not the primary name of the mathematical objects.

Use upper-case $D_k$ for the scalar diagonal element of $D$, matching the companion
manuscript. This keeps the dilation notation visually distinct from the manuscript's other
variables written with lower-case $d$ and superscripts. The package's weighted response
directions use response-by-mode orientation $QD$, with column $k$ equal to $D_kQ_{:k}$. The
manuscript's mode-by-response orientation is its transpose:

\[
DQ^{\mathsf T}=(QD)^{\mathsf T}.
\]

The public API retains the familiar word `n_components`, but for Π-PLS it counts paired latent
modes. It does not count predictor-SVD basis vectors or synthetic latent components. The public
name `predictor_rank` denotes the dimension of the retained observed predictor subspace. Reserve
“predictor signal rank” for synthetic truth where the noiseless generating rank is known.

Recommended manuscript edits are:

- call $P$ and $Q$ predictor and response directions, or orthonormal basis matrices, rather than
  loadings or weight factors;
- replace “projection matrix $P$” with “predictor-direction matrix $P$,” because $P$ itself is not
  a projector;
- use upper-case $D_k$ for diagonal entries of $D$;
- define $h$ as the number of paired latent modes while retaining “components” where a generic
  comparison with PLS or CCA requires shared terminology;
- state explicitly that $DQ^{\mathsf T}$ and package-facing $QD$ are transposed orientations of the
  same weighted response directions.

The terminology is propagated through README material, generated API prose, tutorials, examples,
and public source docstrings. Bibliographic titles remain quoted as published, and historical
change records need not be rewritten solely to modernize spelling.

## Consequences

- The package has one precise vocabulary for the fixed Π-PLS objects and public rank names.
- Predictor and response directions remain clearly distinct from reconstruction loadings.
- Public Python identifiers remain unchanged.
- Manuscript terminology improvements are explicit rather than silently weakened in package prose.
- No estimator, search, validation, synthetic, benchmark, example, rendering, or real-data behavior
  changes.
