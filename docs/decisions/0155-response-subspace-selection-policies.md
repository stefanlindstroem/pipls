# Decision 0155: response-subspace selection policies

## Status

Accepted, implemented, and closed. The peer-reviewed cross-covariance construction remains the
package default. The least-squares-driven response-subspace policy is implemented as a software
extension; it is not part of the companion publication. The six-step implementation and final
release/distribution audit are complete.

## Context

The fixed Pi-PLS construction first forms the retained predictor coordinates

\begin{equation}
\mathbf{Z}=\mathbf{X}\mathbf{\Pi},
\end{equation}

and then chooses an $h$-dimensional orthonormal response basis
$\mathbf{C}\in\mathbb{R}^{q\times h}$. The peer-reviewed companion publication defines this basis
from the dominant right-singular subspace of $\mathbf{Z}^{\mathsf T}\mathbf{Y}$, equivalently by
maximizing retained predictor-response cross-covariance. The package implements that construction
as the default and also provides the software extension defined below.

A second construction is useful for programming users. Instead of choosing $\mathbf{C}$ by the
cross-covariance criterion, one may choose the response subspace that minimizes the rank-$h$
least-squares residual after the predictor projection has been fixed. This is Choice C in the
exploratory derivation that motivated the software extension and is equivalent to reduced-rank
regression of $\mathbf{Y}$ on $\mathbf{Z}$.

The two constructions therefore differ only in how the intermediate response subspace is selected.
The later least-squares coupling, diagonalization, paired predictor/response directions, prediction
map, preprocessing semantics, and model-selection contracts can remain common.

## Decision

### Expose one response-subspace parameter on the fixed estimator

`PiPLSRegression` exposes

```python
response_subspace="cross_covariance"
```

with exactly two accepted public values:

- `"cross_covariance"`: the peer-reviewed Pi-PLS response-subspace construction and package
  default;
- `"least_squares"`: the reduced-rank-regression-inspired software extension defined below.

No abbreviations or compatibility aliases such as `"xcov"` or `"lstsq"` are accepted. The
variance-only response truncation explored separately is not added to the package API.

The default must reproduce the pre-Decision-0155 fixed-estimator numerical path, subject only to
ordinary floating-point behavior. Existing code that does not pass `response_subspace` therefore
continues to fit the peer-reviewed construction.

### Cross-covariance response-subspace policy

Let

\begin{equation}
\mathbf{S}_{\mathrm{ZY}}=\mathbf{Z}^{\mathsf T}\mathbf{Y}.
\end{equation}

The default policy chooses

\begin{equation}
\mathbf{C}_{\mathrm{cov}}
=
\arg\max_{\mathbf{C}^{\mathsf T}\mathbf{C}=\mathbf{I}_h}
\left\|\mathbf{S}_{\mathrm{ZY}}\mathbf{C}\right\|_{\mathrm F}^{2}.
\end{equation}

If

\begin{equation}
\mathbf{S}_{\mathrm{ZY}}
=
\mathbf{U}_{\mathrm{cov}}\mathbf{\Sigma}_{\mathrm{cov}}
\mathbf{V}_{\mathrm{cov}}^{\mathsf T},
\end{equation}

then $\mathbf{C}_{\mathrm{cov}}$ is the span of the first $h$ columns of
$\mathbf{V}_{\mathrm{cov}}$. This remains the canonical construction for manuscript reproduction.

### Least-squares response-subspace policy

The software extension chooses the response subspace and reduced regression map jointly by

\begin{equation}
(\mathbf{C}_{\mathrm{LS}},\mathbf{W}_{\mathrm{LS}})
=
\arg\min_{\substack{\mathbf{C}^{\mathsf T}\mathbf{C}=\mathbf{I}_h\\
\mathbf{W}\in\mathbb{R}^{r_\pi\times h}}}
\left\|\mathbf{Y}-\mathbf{Z}\mathbf{W}\mathbf{C}^{\mathsf T}\right\|_{\mathrm F}^{2}.
\end{equation}

For fixed $\mathbf{C}$, the minimizing map is

\begin{equation}
\mathbf{W}=\mathbf{Z}^{+}\mathbf{Y}\mathbf{C}.
\end{equation}

With the orthogonal projector

\begin{equation}
\mathbf{P}_{\mathbf{Z}}=\mathbf{Z}\mathbf{Z}^{+},
\end{equation}

eliminating $\mathbf{W}$ gives the equivalent response-subspace problem

\begin{equation}
\mathbf{C}_{\mathrm{LS}}
=
\arg\max_{\mathbf{C}^{\mathsf T}\mathbf{C}=\mathbf{I}_h}
\operatorname{tr}\left(
\mathbf{C}^{\mathsf T}\mathbf{Y}^{\mathsf T}
\mathbf{P}_{\mathbf{Z}}\mathbf{Y}\mathbf{C}
\right).
\end{equation}

Equivalently, since

\begin{equation}
\mathbf{Y}^{\mathsf T}\mathbf{P}_{\mathbf{Z}}\mathbf{Y}
=
\mathbf{S}_{\mathrm{ZY}}^{\mathsf T}
(\mathbf{Z}^{\mathsf T}\mathbf{Z})^{+}
\mathbf{S}_{\mathrm{ZY}},
\end{equation}

$\mathbf{C}_{\mathrm{LS}}$ is the dominant eigenspace of the fitted-response cross-product. This is
the rank-$h$ reduced-rank regression response subspace for regression of $\mathbf{Y}$ on the fixed
retained predictor coordinates $\mathbf{Z}$.

The least-squares policy therefore minimizes training squared error over all rank-$h$ maps from
$\mathbf{Z}$ to $\mathbf{Y}$. For the same fixed $(h,r_\pi)$, its training residual cannot exceed
the residual obtained from any other admissible response basis, including the cross-covariance
basis, apart from numerical tolerance. This optimization property does not imply uniformly lower
cross-validated or external prediction error.

### Share the downstream Pi-PLS construction

Once either response basis $\mathbf{C}$ has been selected, both policies use the same remaining
construction:

\begin{equation}
\mathbf{W}=\mathbf{Z}^{+}\mathbf{Y}\mathbf{C},
\end{equation}

followed by

\begin{equation}
\mathbf{W}=\mathbf{M}\mathbf{D}\mathbf{N}^{\mathsf T},
\qquad
\mathbf{P}=\mathbf{\Pi}\mathbf{M},
\qquad
\mathbf{Q}=\mathbf{C}\mathbf{N}.
\end{equation}

Thus the fitted relation remains

\begin{equation}
\widehat{\mathbf{Y}}=\mathbf{X}\mathbf{P}\mathbf{D}\mathbf{Q}^{\mathsf T}
\end{equation}

in centered/scaled coordinates. The alternative policy changes the construction of
$\mathbf{C}$, not the meaning of the final paired directions $\mathbf{P}$ and $\mathbf{Q}$ or the
diagonal dilation $\mathbf{D}$.

### Use a stable exact numerical construction for the least-squares policy

The package implementation must not form
$\mathbf{S}_{\mathrm{ZY}}^{\mathsf T}(\mathbf{Z}^{\mathsf T}\mathbf{Z})^{+}
\mathbf{S}_{\mathrm{ZY}}$ through explicit normal equations with a new hard-coded pseudoinverse
cutoff. That route squares the condition number and would introduce a numerical-rank convention
separate from the package's predictor-rank policy.

Instead, the implementation obtains an orthonormal basis $\mathbf{U}_{\mathbf{Z}}$ for
$\operatorname{col}(\mathbf{Z})$ through a stable exact orthogonal factorization, such as reduced QR
or an equivalent exact SVD. Since

\begin{equation}
\mathbf{P}_{\mathbf{Z}}
=
\mathbf{U}_{\mathbf{Z}}\mathbf{U}_{\mathbf{Z}}^{\mathsf T},
\end{equation}

we have

\begin{equation}
\mathbf{Y}^{\mathsf T}\mathbf{P}_{\mathbf{Z}}\mathbf{Y}
=
(\mathbf{U}_{\mathbf{Z}}^{\mathsf T}\mathbf{Y})^{\mathsf T}
(\mathbf{U}_{\mathbf{Z}}^{\mathsf T}\mathbf{Y}).
\end{equation}

Therefore $\mathbf{C}_{\mathrm{LS}}$ can be obtained as the dominant right-singular subspace of
$\mathbf{U}_{\mathbf{Z}}^{\mathsf T}\mathbf{Y}$ without forming a Gram-matrix pseudoinverse.

Response-subspace selection remains exact for both policies. `svd_solver` continues to govern only
the predictor decomposition that constructs $\mathbf{\Pi}$. A randomized predictor SVD must not
silently make either response-subspace factorization randomized.

### Keep response-subspace policy outside the search dimensions

`response_subspace` is configuration of the fixed `PiPLSRegression` estimator. It is not a third
candidate dimension of `PiPLSSearchCV`.

`PiPLSSearchCV` continues to search only component count $h$ and predictor rank $r_\pi$ and must
preserve the configured response-subspace policy when cloning its estimator template for candidate
fits, OOF prediction, and final refitting. The package does not automatically compare
`"cross_covariance"` and `"least_squares"` within one search object.

A programming user who wants to compare the two policies should perform two searches using two
estimator templates and, for a controlled comparison, the same materialized CV splits.

### Preserve the publication boundary explicitly

The peer-reviewed companion publication contains only the cross-covariance response-subspace
construction. `"least_squares"` is a package extension motivated by reduced-rank regression and is
not part of that publication.

The source implementation and the theory documentation for the least-squares policy must contain an
explicit note to that effect. Manuscript-reproduction documentation and maintained publication-
aligned examples continue to use `"cross_covariance"`. Documentation may state that the
least-squares policy can be useful for some problems, but must not present it as peer-reviewed
Pi-PLS evidence or claim that it is uniformly superior in predictive performance.

## Relationship to earlier decisions

This decision extends the implemented runtime contracts recorded by earlier current decisions.
The following reconciliations define their continuing scope:

- Decision 0120 continues to define the companion-manuscript cross-covariance construction and its
  publication scope; Decision 0155 adds a software-only alternative without changing what the
  manuscript contains.
- Decision 0008 continues to allow randomized decomposition only for the predictor basis. Its
  exact response-side factorization contract applies to whichever response-subspace construction is
  active.
- Decision 0039 continues to separate fixed-estimator fitting from search ownership. The
  response-subspace policy belongs to the fixed estimator and is propagated, not optimized, by
  search.

The fixed estimator now implements both response-subspace policies, and search propagates the
configured policy without optimizing it. User-facing theory and API guidance document both
policies, while publication-aligned material continues to identify the cross-covariance
construction as the peer-reviewed method. The maintained programming example compares the
policies with two searches on the same materialized CV splits and labels the resulting CV-MSE as
model-development evidence.

## Implementation sequence

This decision was implemented through a six-step migration:

1. establish the decision and maintainer contract;
2. implement the fixed numerical core for both response-subspace policies;
3. expose and propagate the public estimator configuration through search and refitting;
4. add mathematical, numerical, and API regression coverage;
5. document the theory and API and add a programming-user comparison example;
6. complete the release, stale-contract, installed-artifact, and distribution audit.

Step 1 is split into reviewable patches 1A--1D and is complete. Patch 1A recorded Decision 0155
and indexed it; Patch 1B reconciled the affected earlier decisions; Patch 1C opened the active
maintainer roadmap; and Patch 1D audited repository-wide consistency before runtime implementation.
Step 2 is split into patches 2A--2D and is also complete: the core isolates the published
cross-covariance response basis, implements the exact least-squares/Choice-C basis, dispatches
between exactly the two accepted private policies, and protects both the randomized-predictor and
`p >> n` numerical regimes. Step 3 is complete in patches 3A--3C: `PiPLSRegression` now exposes
`response_subspace` with `"cross_covariance"` as the default, search/OOF/refit/pipeline paths preserve
the configured policy without searching it, and the implemented public contract is synchronized.
Step 4 is complete in patches 4A--4C: the repository now protects an independent Choice-C
reference, direct RRR equivalence, least-squares training optimality, limiting-case identities,
shared factorization invariants, scaling/serialization/interoperability behavior, and synchronized
internal mathematics/testing contracts. Step 5 is complete in patches 5A--5C: user-facing
theory, API/workflow guidance, manuscript-alignment guidance, and the matched-split programming
comparison are maintained. Step 6 completed the changelog/stale-contract audit, strengthened
clean installed-artifact qualification for both policies, exercised the then-current dedicated
response-subspace comparison from the source distribution, and completed the final repository
audit. Documentation-from-sdist qualification
builds from the extracted source tree with an explicit artifact `PYTHONPATH`; clean installation
isolation remains owned by the wheel/sdist distribution check.

## Subsequent refinement

The dedicated Pulp-only response-subspace comparison was later consolidated into Example 03. The
scientific and API contracts of this decision are unchanged: the two policies remain separate
fixed-estimator configurations, and the least-squares policy remains outside the peer-reviewed
publication. The maintained comparison presents both policies alongside ordinary PLS on matched
folds for Pulp, Sugarcane, and Tobacco.

## Consequences

- Existing users retain the peer-reviewed cross-covariance construction by default.
- Programming users gain an explicit least-squares-driven response-subspace option without changing
  predictor-rank or component-count semantics.
- Both policies retain the same final diagonal paired-mode representation.
- The least-squares policy has a direct reduced-rank-regression interpretation after the panoramic
  predictor projection.
- Comparing the two response policies remains an explicit modeling decision rather than an
  additional hidden search axis.
- The publication/software distinction is part of the package contract and must remain visible in
  source and documentation.
