# Pi-PLS theory reference

## Purpose and authority

This document is the persistent conceptual reference for LLM-assisted development of Pi-PLS.
It summarizes the mathematical construction, interpretation, limiting cases, rank roles, and
implementation consequences of the method so that the project owner does not need to attach the
manuscript for routine implementation work.

The companion manuscript, “Panoramic Partial Least Squares (Pi-PLS): Transparent, parsimonious,
and more interpretable multivariate regression model,” under revision at *Computers & Chemical
Engineering* as CACE-D-26-00847, is the scientific source for the canonical cross-covariance
construction summarized here. Decision 0155 additionally defines the implemented least-squares/
RRR-inspired response-subspace policy as a software extension that is not part of the peer-reviewed
publication. This file is a repository-maintained explanation of the implemented scientific
contract, not a replacement for the manuscript. The authority hierarchy for implementation work is:

1. explicit scientific decisions from the project owner;
2. accepted decision records under `docs/decisions/`;
3. the normative equations and invariants in `docs/maintainers/mathematics.md`;
4. the numerical rules in `docs/maintainers/numerical_contracts.md`;
5. this conceptual reference;
6. source code and tests as evidence of currently implemented behavior.

If a future manuscript revision conflicts with this document, do not silently reconcile the two.
Record the discrepancy, obtain an explicit decision, and update all affected contracts together.

## Problem setting

Let

- $\mathbf{X}\in\mathbb{R}^{n\times p}$ be the predictor matrix;
- $\mathbf{Y}\in\mathbb{R}^{n\times q}$ be the response matrix;
- $n$ be the number of observations;
- $p$ be the number of predictor variables;
- $q$ be the number of response variables.

The theoretical core is formulated for column-centered matrices. The public estimator may also
scale columns, but centering and scaling are preprocessing operations outside the fixed-parameter
core. In centered or centered-and-scaled coordinates, multivariate linear regression seeks
$\mathbf{B}\in\mathbb{R}^{p\times q}$ such that

$$
\mathbf{Y}=\mathbf{X}\mathbf{B}+\mathbf{E}.
$$

Unconstrained multivariate OLS uses the minimum-norm coefficient matrix
$\mathbf{B}_{\mathrm{OLS}}=\mathbf{X}^{+}\mathbf{Y}$. Pi-PLS instead separates two structural
choices:

1. which predictor subspace is retained, controlled by the predictor rank $r_\pi$;
2. how many coupled predictor-response modes are retained, controlled by the latent dimension
   $h$.

This separation is central. The roles of $r_\pi$ and $h$ are not interchangeable.

## Conceptual position among related methods

Pi-PLS combines ideas familiar from several multivariate methods:

- Like truncated-SVD regression, it first restricts estimation to a rank-controlled predictor
  subspace.
- In its default peer-reviewed response-subspace policy, like PLS and PLS-SVD, it uses
  predictor-response cross-covariance to identify response-relevant latent directions.
- Its optional least-squares response-subspace policy applies the reduced-rank-regression principle
  after the predictor subspace has been fixed.
- Like reduced-rank regression, both policies represent the coefficient matrix through a low-rank
  factorization.
- Unlike iterative deflation-based PLS algorithms, the fixed-parameter Pi-PLS construction is a
  sequence of orthogonal factorizations and least-squares operations with an explicit closed form.
- Unlike PLS-SVD, Pi-PLS first preserves a chosen rank-controlled predictor subspace and only then
  selects the response subspace under the configured criterion.

The defining interpretive feature is a diagonal one-to-one coupling between orthogonal predictor
and response modes.

## Notation and dimensions

| Quantity | Shape | Meaning |
|---|---:|---|
| $\mathbf{X}$ | $(n,p)$ | centered or centered-and-scaled predictors supplied to the core |
| $\mathbf{Y}$ | $(n,q)$ | centered or centered-and-scaled responses supplied to the core |
| $r_\pi$ | scalar | retained predictor-subspace dimension |
| $h$ | scalar | number of coupled latent modes |
| $\mathbf{\Pi}$ | $(p,r_\pi)$ | orthonormal retained predictor basis |
| $\mathbf{Z}=\mathbf{X}\mathbf{\Pi}$ | $(n,r_\pi)$ | predictor scores in the retained subspace |
| $\mathbf{C}$ | $(q,h)$ | orthonormal response basis selected by the configured response-subspace policy |
| $\mathbf{W}$ | $(r_\pi,h)$ | least-squares map from $\mathbf{Z}$ to $\mathbf{Y}\mathbf{C}$ |
| $\mathbf{M}$ | $(r_\pi,h)$ | left singular vectors of $\mathbf{W}$ |
| $\mathbf{D}$ | $(h,h)$ | nonnegative diagonal dilation matrix |
| $\mathbf{N}$ | $(h,h)$ | right singular vectors of $\mathbf{W}$ |
| $\mathbf{P}=\mathbf{\Pi}\mathbf{M}$ | $(p,h)$ | orthonormal predictor directions |
| $\mathbf{Q}=\mathbf{C}\mathbf{N}$ | $(q,h)$ | orthonormal response directions |
| $\mathbf{B}_{\mathrm{cs}}$ | $(p,q)$ | regression map in core coordinates |

The dimensional admissibility conditions are

$$
1\le h\le \min(r_\pi,q),
\qquad
h\le r_\pi\le \min(n,p).
$$

The implementation additionally requires $r_\pi$ not to exceed the numerical rank of the supplied
predictor matrix.

## Canonical terminology

Use these terms in mathematical and explanatory prose:

- $\mathbf{\Pi}$ is the **retained predictor basis**;
- $\mathbf{\Pi}\mathbf{\Pi}^{\mathsf T}$ is the **retained-subspace projector**;
- $\mathbf{P}$ and $\mathbf{Q}$ contain **orthonormal predictor directions** and
  **orthonormal response directions**;
- $D_k=D_{kk}$ is the **dilation** of paired latent mode $k$;
- $\mathbf{X}\mathbf{P}$ and $\mathbf{Y}\mathbf{Q}$ are the **predictor scores** and
  **response scores**;
- $(P_{:k},D_k,Q_{:k})$ is **paired latent mode** $k$;
- public `n_components` is the number of paired latent modes $h$;
- public `predictor_rank` is the retained predictor-subspace dimension $r_\pi$.

Do not call $P$ a projection matrix: $PP^{\mathsf T}$ is the projector onto the final predictor-
direction span. Do not call $P$ or $Q$ reconstruction loadings: the estimator's `x_loadings_` and
`y_loadings_` are separate least-squares reconstruction quantities. `PiPLSDecomposition` exposes
`predictor_directions` and `response_directions`. The
estimator retains the standard PLS-style names `x_rotations_` and `y_rotations_`, but **direction**
is the canonical mathematical term.

The package's response-by-mode weighted directions are $QD$, with column $k$ equal to $D_kQ_{:k}$.
The mode-by-response form used in parts of the manuscript is exactly

$$
DQ^{\mathsf T}=(QD)^{\mathsf T}.
$$

Reserve “predictor signal rank” for synthetic truth where the noiseless rank is known. For observed
data, use retained predictor basis, retained predictor subspace, and predictor rank.

## Step 1: retain a rank-controlled predictor subspace

Let the singular value decomposition of the core predictor matrix be

$$
\mathbf{X}=\mathbf{U}_{\mathrm{X}}\mathbf{S}_{\mathrm{X}}\mathbf{V}_{\mathrm{X}}^{\mathsf T}.
$$

Retain the leading $r_\pi$ right singular vectors:

$$
\mathbf{\Pi}=\mathbf{V}_{\mathrm{X}}(:,1:r_\pi),
\qquad
\mathbf{\Pi}^{\mathsf T}\mathbf{\Pi}=\mathbf{I}_{r_\pi}.
$$

The corresponding orthogonal projector is
$\mathbf{\Pi}\mathbf{\Pi}^{\mathsf T}$, and the predictor matrix decomposes as

$$
\mathbf{X}
=
\mathbf{X}\mathbf{\Pi}\mathbf{\Pi}^{\mathsf T}
+
\mathbf{X}(\mathbf{I}_p-\mathbf{\Pi}\mathbf{\Pi}^{\mathsf T}).
$$

Define the retained predictor scores

$$
\mathbf{Z}=\mathbf{X}\mathbf{\Pi}.
$$

The first term in the decomposition is the rank-$r_\pi$ predictor approximation
$\mathbf{Z}\mathbf{\Pi}^{\mathsf T}$. The second term is orthogonal to the retained predictor
subspace and is treated as truncation residual. Thus, $r_\pi$ determines which predictor variation
remains available to all later stages. Any predictive direction removed here cannot be recovered
by increasing $h$.

## Step 2: select the response subspace

The package implements two response-subspace policies. Both produce an orthonormal basis
$\mathbf{C}\in\mathbb{R}^{q\times h}$ and then share the same latent least-squares and
final-diagonalization stages.

### Cross-covariance policy

The default `"cross_covariance"` policy is the construction in the peer-reviewed companion
publication. Within the retained predictor space, form

$$
\boldsymbol{\Sigma}_{\mathrm{ZY}}=\mathbf{Z}^{\mathsf T}\mathbf{Y}
\in\mathbb{R}^{r_\pi\times q}.
$$

The response basis maximizes retained squared cross-covariance:

$$
\max_{\mathbf{C}^{\mathsf T}\mathbf{C}=\mathbf{I}_h}
\left\|\boldsymbol{\Sigma}_{\mathrm{ZY}}\mathbf{C}\right\|_{\mathrm{F}}^2.
$$

If

$$
\boldsymbol{\Sigma}_{\mathrm{ZY}}=\mathbf{U}\mathbf{S}\mathbf{V}^{\mathsf T},
$$

then one optimum is

$$
\mathbf{C}_{\mathrm{cov}}=\mathbf{V}_{(:,1:h)}.
$$

Thus $\mathbf{C}_{\mathrm{cov}}$ spans the $h$ response directions most strongly coupled, in the
cross-covariance sense, to the retained predictor representation.

### Least-squares policy

The optional `"least_squares"` policy is an RRR-inspired software extension and is **not part of
the peer-reviewed companion publication**. It chooses the response subspace and reduced map jointly:

$$
(\mathbf{C}_{\mathrm{LS}},\mathbf{W}_{\mathrm{LS}})
=
\arg\min_{\substack{\mathbf{C}^{\mathsf T}\mathbf{C}=\mathbf{I}_h\\
\mathbf{W}\in\mathbb{R}^{r_\pi\times h}}}
\left\|\mathbf{Y}-\mathbf{Z}\mathbf{W}\mathbf{C}^{\mathsf T}\right\|_{\mathrm{F}}^2.
$$

For fixed $\mathbf{C}$, the minimizing map is

$$
\mathbf{W}=\mathbf{Z}^{+}\mathbf{Y}\mathbf{C}.
$$

With the projector $\mathbf{P}_{\mathbf{Z}}=\mathbf{Z}\mathbf{Z}^{+}$, eliminating
$\mathbf{W}$ gives

$$
\mathbf{C}_{\mathrm{LS}}
=
\arg\max_{\mathbf{C}^{\mathsf T}\mathbf{C}=\mathbf{I}_h}
\operatorname{tr}\left[
\mathbf{C}^{\mathsf T}\mathbf{Y}^{\mathsf T}
\mathbf{P}_{\mathbf{Z}}\mathbf{Y}\mathbf{C}
\right].
$$

Equivalently, the least-squares response subspace is the dominant eigenspace of
$\mathbf{Y}^{\mathsf T}\mathbf{P}_{\mathbf{Z}}\mathbf{Y}$. It is therefore the rank-$h$
reduced-rank-regression response subspace for regression of $\mathbf{Y}$ on the fixed retained
predictor coordinates $\mathbf{Z}$. The implementation obtains it by exact reduced QR of
$\mathbf{Z}$ followed by exact SVD of $\mathbf{Q}_{\mathbf{Z}}^{\mathsf T}\mathbf{Y}$; it does
not form normal equations or introduce a separate pseudoinverse cutoff.

For the same fixed $(h,r_\pi)$, the least-squares policy cannot have larger training residual norm
than the cross-covariance policy, apart from numerical tolerance. This is a training-objective
statement, not a claim of uniformly lower cross-validated or external prediction error.

For $q=1$, both policies produce the same fitted regression map. They also agree when the complete
response space is retained with $h=q\le r_\pi$. In general, basis vectors are not unique when the
relevant singular or eigenvalues are repeated; the selected subspace and fitted map are the
meaningful objects.

## Step 3: regress in the reduced latent coordinates

Project the response matrix onto the selected response subspace and solve

$$
\mathbf{Y}\mathbf{C}=\mathbf{Z}\mathbf{W}+\mathbf{E}'',
$$

by least squares. The minimum-norm solution is

$$
\mathbf{W}=\mathbf{Z}^{+}\mathbf{Y}\mathbf{C}.
$$

This step estimates how retained predictor scores map to the selected response scores. It avoids
forming an explicit inverse; the implementation uses a least-squares solver.

The regression map before diagonalization is

$$
\mathbf{B}_{\mathrm{cs}}
=
\mathbf{\Pi}\mathbf{W}\mathbf{C}^{\mathsf T}.
$$

The fitted core-coordinate responses are therefore

$$
\widehat{\mathbf{Y}}
=
\mathbf{X}\mathbf{B}_{\mathrm{cs}}
=
\mathbf{Z}\mathbf{W}\mathbf{C}^{\mathsf T}.
$$

These two prediction expressions must remain numerically equivalent.

## Step 4: diagonalize the latent coupling

Take the economical SVD of the latent regression matrix:

$$
\mathbf{W}=\mathbf{M}\mathbf{D}\mathbf{N}^{\mathsf T},
$$

where

$$
\mathbf{M}^{\mathsf T}\mathbf{M}=\mathbf{I}_h,
\qquad
\mathbf{N}^{\mathsf T}\mathbf{N}=\mathbf{I}_h,
$$

and

$$
\mathbf{D}=\operatorname{diag}(D_1,\ldots,D_h),
\qquad
D_1\ge D_2\ge\cdots\ge D_h\ge0.
$$

Define

$$
\mathbf{P}=\mathbf{\Pi}\mathbf{M},
\qquad
\mathbf{Q}=\mathbf{C}\mathbf{N}.
$$

Because both factors are products of matrices with orthonormal columns,

$$
\mathbf{P}^{\mathsf T}\mathbf{P}=\mathbf{I}_h,
\qquad
\mathbf{Q}^{\mathsf T}\mathbf{Q}=\mathbf{I}_h.
$$

The projected regression relation becomes

$$
\mathbf{Y}\mathbf{Q}
=
\mathbf{X}\mathbf{P}\mathbf{D}
+
\mathbf{E}_{\pi},
$$

with $\mathbf{E}_{\pi}=\mathbf{E}''\mathbf{N}$. Orthogonal rotation preserves the residual
Frobenius norm and rotates the residual covariance without changing its eigenvalues.

This is the panoramic one-to-one representation. Column $j$ of $\mathbf{X}\mathbf{P}$ is coupled
only to column $j$ of $\mathbf{Y}\mathbf{Q}$, with dilation $d_j$. There are no
cross-coupling terms between distinct latent modes in these coordinates.

## Why the method is panoramic

Unlike standard deflation-based PLS algorithms, Pi-PLS fixes one rank-controlled predictor
representation $\mathbf{Z}=\mathbf{X}\mathbf{\Pi}$ and derives all $h$ coupled modes from that
undeflated retained space. The full retained predictor subspace remains available during
response-subspace selection and the latent least-squares stage; in this sense, the method is
panoramic.

## Regression-map identities and prediction

The centered/scaled regression map has two equivalent factorizations:

$$
\boxed{
\mathbf{B}_{\mathrm{cs}}
=
\mathbf{P}\mathbf{D}\mathbf{Q}^{\mathsf T}
=
\mathbf{\Pi}\mathbf{W}\mathbf{C}^{\mathsf T}
}
$$

and hence

$$
\widehat{\mathbf{Y}}
=
\mathbf{X}\mathbf{P}\mathbf{D}\mathbf{Q}^{\mathsf T}.
$$

For a new core-coordinate predictor matrix $\mathbf{X}_{\mathrm{new}}$,

$$
\widehat{\mathbf{Y}}_{\mathrm{new}}
=
\mathbf{X}_{\mathrm{new}}\mathbf{P}\mathbf{D}\mathbf{Q}^{\mathsf T}.
$$

In the public estimator, let $\boldsymbol\mu_{\mathrm{X}}$ and
$\boldsymbol\mu_{\mathrm{Y}}$ be training means, and let
$\mathbf{s}_{\mathrm{X}}$, $\mathbf{s}_{\mathrm{Y}}$ be the safe training scales. Then

$$
\mathbf{X}_{\mathrm{cs}}
=
(\mathbf{X}-\boldsymbol\mu_{\mathrm{X}})\operatorname{diag}(\mathbf{s}_{\mathrm{X}})^{-1},
$$

and predictions are transformed back to response units. Equivalently, the original-unit
coefficient matrix is

$$
\mathbf{B}
=
\operatorname{diag}(\mathbf{s}_{\mathrm{X}})^{-1}
\mathbf{B}_{\mathrm{cs}}
\operatorname{diag}(\mathbf{s}_{\mathrm{Y}}),
$$

with intercept

$$
\mathbf{b}_0
=
\boldsymbol\mu_{\mathrm{Y}}-\boldsymbol\mu_{\mathrm{X}}\mathbf{B}.
$$

Each scale vector is learned or set to ones independently. `scale` supplies the default for both
blocks, while non-`None` `scale_x` and `scale_y` values override predictor and response scaling.
If scaling is disabled for one block, the corresponding expression reduces to centering only.

## Interpretation of the two ranks

### Predictor rank $r_\pi$

$r_\pi$ controls how much predictor-side variation is retained before response information is
used. It is a regularization and subspace-retention parameter.

- If $r_\pi$ is too small, predictive directions may be irreversibly removed.
- Once all predictive directions are retained, additional predictor directions may have little
  effect because response-subspace selection and the reduced regression can give them negligible
  contribution to the fitted map.
- This creates an important asymmetry: under-specification can destroy signal, whereas moderate
  over-specification may mainly increase computation or variance.

The manuscript reports an empirical sharp-decrease-then-plateau pattern in CV error as $r_\pi$
increases. This pattern motivates practical upper bounds and adaptive search, but it is not a
proof that every dataset has a unimodal or perfectly flat CV curve.

### Latent dimension $h$

$h$ controls the rank and interpretive complexity of the fitted predictor-response coupling.
It is the number of paired orthogonal modes retained after the predictor subspace has been fixed.
Increasing $h$ cannot recover predictor directions excluded by $r_\pi$.

The admissible relation $h\le r_\pi$ is structural: the latent regression matrix
$\mathbf{W}\in\mathbb{R}^{r_\pi\times h}$ cannot support more independent coupled modes than the
retained predictor dimension. Also $h\le q$ because the response latent basis has $q$ rows.

## Model size

After $\mathbf{\Pi}$ has been fixed, the fitted representation is described by
$\mathbf{M}$, $\mathbf{D}$, and $\mathbf{Q}$. They contribute $r_\pi h$, $h$, and $qh$
entries. The orthonormality constraints on $\mathbf{M}$ and $\mathbf{Q}$ each remove
$h(h+1)/2$ degrees of freedom. The nominal fitted dimension is therefore

$$
\boxed{(r_\pi+q-h)h}.
$$

This is the internally consistent count derived in Section 3.3 of the companion manuscript. It
does not treat the $ph$ entries of $\mathbf{P}=\mathbf{\Pi}\mathbf{M}$ as independently free
parameters. The later manuscript discussion uses an incomplete shorthand; the explicit derivation
is normative for this repository.

## Limiting and comparative cases

### OLS limit

If the predictor truncation preserves the estimable predictor row space and $h$ retains every
estimable response-side direction, then

$$
\widehat{\mathbf{Y}}
=
\mathbf{Z}\mathbf{W}\mathbf{C}^{\mathsf T}
$$

coincides with the multivariate OLS fitted response. In that rank-preserving case, Pi-PLS is an
orthogonal latent-coordinate representation of the same fitted map rather than a different fit.

### Relation to reduced-rank regression

Both methods produce low-rank coefficient structures. Under the default cross-covariance policy,
Pi-PLS first selects a rank-controlled predictor subspace and then chooses response directions from
$\mathbf{Z}^{\mathsf T}\mathbf{Y}$ before rotating the latent regression into a diagonal one-to-one
coupling. Under `"least_squares"`, the response-subspace step is exactly the rank-$h$ reduced-rank-
regression solution for $\mathbf{Y}$ on the fixed retained coordinates $\mathbf{Z}$; Pi-PLS then
applies the same final diagonal pairing $\mathbf{P}$, $\mathbf{D}$, $\mathbf{Q}$ used by the default
construction.

### Relation to CCA

CCA also constructs paired predictor and response variates with a diagonal association structure,
but classical CCA maximizes normalized correlation after within-block whitening. The peer-reviewed
Pi-PLS policy uses an unwhitened cross-covariance criterion within the retained predictor
representation and then estimates a predictive least-squares map. The optional software extension
instead selects the response subspace by a least-squares/RRR criterion. In either case, the final
diagonal relation is structurally analogous to CCA, not identical to its objective.

### Relation to PLS and PLS-SVD

PLS and the default Pi-PLS response policy both exploit predictor-response covariance. Classical
PLS is usually constructed through iterative component extraction and deflation. Pi-PLS instead
uses a closed sequence of orthogonal factorizations and least-squares operations.

PLS-SVD derives both sides directly from a cross-covariance operator. Pi-PLS first fixes the
retained predictor subspace from the singular structure of $\mathbf{X}$ and then selects the
response subspace under the configured criterion. Under the peer-reviewed default this criterion is
cross-covariance with the retained subspace; under the software extension it is the retained-space
least-squares/RRR criterion. The predictor truncation can therefore preserve variation not aligned
with the leading raw cross-covariance modes.

## Identifiability and invariance

Individual singular vectors are not generally identifiable mathematical objects.

- Every singular-vector pair is sign-indeterminate.
- Repeated singular values permit arbitrary orthogonal rotations within the corresponding
  invariant subspace.
- Nearly repeated singular values can make individual columns numerically unstable even when the
  subspace and regression map are stable.
- A zero dilation value makes its associated paired directions irrelevant to prediction and
  generally non-identifiable.

Consequently, scientific and regression tests should compare:

- projection matrices such as $\mathbf{P}\mathbf{P}^{\mathsf T}$;
- principal angles between subspaces;
- ordered singular or dilation values;
- the regression map $\mathbf{B}_{\mathrm{cs}}$;
- fitted values and predictions;
- reconstruction identities.

Tests should not require raw singular-vector columns to match a fixed sign or basis orientation.

## Numerical-rank and degeneracy policy

The mathematical bound $r_\pi\le\min(n,p)$ is insufficient for finite-precision computation. The
private core estimates predictor numerical rank from the thin-SVD singular values $s_i$ using

$$
\tau_{\mathrm{X}}=\max(n,p)\,\epsilon_{64}\,s_1,
$$

and counts singular values satisfying $s_i>\tau_{\mathrm{X}}$. A requested $r_\pi$ above this
numerical rank
is invalid and raises an error rather than being silently clamped.

Constant columns, rank-deficient matrices, $p\gg n$, singleton CV training folds, and repeated
singular values are expected boundary cases. Their exact handling is specified in
`docs/maintainers/numerical_contracts.md`.

## Predictor-rank selection: theory versus API policy

The manuscript discusses $r_\pi$ as the dimension that must be large enough to retain the full
predictor-response coupling. It reports that under-specification causes irreversible loss, while
moderate over-specification often lies on a broad predictive plateau. It also discusses
sample-size heuristics and exhaustive CV as practical selection approaches.

The repository API separates hard predictor-rank feasibility from optional rank policies. For
internal CV, the hard/default upper bound is

\begin{equation}
r_{\pi,\mathrm{hard}}
=
\min\left(
 p_{\mathrm{min}},
 n_{\mathrm{train,min}}-1,
 r_{\mathrm{num,min}}
\right),
\end{equation}

where $p_{\mathrm{min}}$ is the minimum predictor count after fold-local pipeline preprocessing,
$n_{\mathrm{train,min}}$ is the smallest materialized training-fold size, and
$r_{\mathrm{num,min}}$ is the minimum predictor rank verified after terminal-estimator centering
and optional scaling. With `max_predictor_rank=None`, automatic search uses the complete
hard-feasible domain; an explicit integer maximum only restricts that domain further.

The EPV-inspired policy is explicit rather than a general search ceiling. With
`predictor_rank_values="epv"`, the nominal full-sample rank is

\begin{equation}
r_{\pi,\mathrm{epv}}
=
\min\left(
 p,
 \left\lceil
 \frac{n}
 {\texttt{samples\_per\_predictor\_rank}}
 \right\rceil
\right),
\end{equation}

and that fixed rank is clipped only by hard feasibility and any explicit integer maximum. The
samples-per-rank quantity does not cap ordinary automatic or exhaustive search.

The fixed estimator accepts one explicit integer $r_\pi$. Full-domain exhaustive search, adaptive
candidate coverage, explicit rank sequences, and EPV belong to `PiPLSSearchCV`, which evaluates
fixed-estimator clones on the admissible triangular surface.

Adaptive search reports every evaluated rank and does not guarantee the exhaustive optimum for
an arbitrary non-unimodal CV curve. Search approximation and linear-algebra approximation are
separate. The implemented `svd_solver` policy permits randomized approximation only for the first
predictor-matrix SVD; response-side factorizations and the coupling SVD remain exact. Automatic
solver selection is conservative and depends on matrix dimensions and retained-rank fraction, while
explicit `"full"` remains the reference path. Response-subspace selection remains exact under
both policies: cross-covariance uses exact SVD of $\mathbf{Z}^{\mathsf T}\mathbf{Y}$, while the
least-squares policy uses exact reduced QR of $\mathbf{Z}$ followed by exact SVD of
$\mathbf{Q}_{\mathbf{Z}}^{\mathsf T}\mathbf{Y}$.

## Cross-validation consequences

Any CV-based selection of $r_\pi$ or $h$ must obey the following theoretical separation:

1. fit centering and scaling only on the training fold;
2. derive predictor and response subspaces only from the training fold;
3. predict the validation fold using that fold-trained model;
4. reuse the same materialized split set for every candidate being compared;
5. define the ordinary automatic rank domain only from fold-local hard feasibility and any
   explicit integer user cap; apply the full-sample $n/c$ rule only when the EPV policy is requested;
6. refit the selected fixed-parameter model once on all data supplied to `fit()`.

Response-standardized MSE is used so response columns with different physical scales contribute
comparably. The scale for each response must be estimated from the corresponding training fold,
not from the validation fold or the complete dataset.

## Synthetic-data capability

Decision 0119 defines one public synthetic-data utility. `make_synthetic_data()` implements the
companion manuscript model exactly:

\begin{equation}
\mathbf{X}
=
\boldsymbol{\Lambda}_{\mathrm{p}}\mathbf{L}_{\mathrm{p}}
+
\boldsymbol{\Lambda}_{\mathrm{s}}\mathbf{L}_{\mathrm{sp}}
+
\boldsymbol{\varepsilon}_{\mathrm{X}},
\qquad
\mathbf{Y}
=
\boldsymbol{\Lambda}_{\mathrm{s}}\mathbf{L}_{\mathrm{sr}}
+
\boldsymbol{\Lambda}_{\mathrm{r}}\mathbf{L}_{\mathrm{r}}
+
\boldsymbol{\varepsilon}_{\mathrm{Y}}.
\end{equation}

All score and loading entries are independent standard-normal draws, and the noise blocks are
independent Gaussian draws with caller-specified standard deviations. No centering,
standardization, orthonormalization, strength scaling, or observed scaling is applied by that
generator. This capability does not change estimator, search, validation, or real-data policy.

## Core algorithm summary

For centered or centered-and-scaled $\mathbf{X}$ and $\mathbf{Y}$:

1. compute a thin SVD of $\mathbf{X}$;
2. set $\mathbf{\Pi}$ to the leading $r_\pi$ right singular vectors;
3. form $\mathbf{Z}=\mathbf{X}\mathbf{\Pi}$;
4. compute a thin SVD of $\mathbf{Z}^{\mathsf T}\mathbf{Y}$;
5. set $\mathbf{C}$ to its leading $h$ right singular vectors;
6. solve $\mathbf{Z}\mathbf{W}\approx\mathbf{Y}\mathbf{C}$ by least squares;
7. compute the economical SVD $\mathbf{W}=\mathbf{M}\mathbf{D}\mathbf{N}^{\mathsf T}$;
8. set $\mathbf{P}=\mathbf{\Pi}\mathbf{M}$ and $\mathbf{Q}=\mathbf{C}\mathbf{N}$;
9. return the factorization and regression map
   $\mathbf{B}_{\mathrm{cs}}=\mathbf{P}\mathbf{D}\mathbf{Q}^{\mathsf T}$.

## Implementation review checklist

A change to the core theory or its implementation should preserve or explicitly revise all of the
following:

- $1\le h\le\min(r_\pi,q)$;
- $r_\pi$ does not exceed predictor numerical rank;
- $\mathbf{\Pi}^{\mathsf T}\mathbf{\Pi}=\mathbf{I}$;
- $\mathbf{C}^{\mathsf T}\mathbf{C}=\mathbf{I}$;
- $\mathbf{P}^{\mathsf T}\mathbf{P}=\mathbf{I}$;
- $\mathbf{Q}^{\mathsf T}\mathbf{Q}=\mathbf{I}$;
- $D_1\ge\cdots\ge D_h\ge0$;
- $\mathbf{B}_{\mathrm{cs}}=\mathbf{P}\mathbf{D}\mathbf{Q}^{\mathsf T}$;
- $\mathbf{B}_{\mathrm{cs}}=\mathbf{\Pi}\mathbf{W}\mathbf{C}^{\mathsf T}$;
- predictions from the two factorizations agree;
- singular-vector signs and repeated-subspace bases are not treated as identifiers;
- preprocessing used in selection is fit within each training fold;
- rank-search approximation is not conflated with SVD approximation.

## Scope boundaries

This file explains the linear Pi-PLS method currently targeted by the repository. It does not
specify nonlinear kernels, sparsity penalties, missing-data algorithms, probabilistic inference,
uncertainty intervals, or causal interpretation. Such extensions require separate scientific
decisions and must not be inferred from the present factorization.
