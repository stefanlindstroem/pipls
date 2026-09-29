# Mathematical contract

This file is the concise normative mathematical contract for implementation. Source code and
tests must be kept consistent with it. Read `docs/maintainers/theory.md` for the fuller derivation,
interpretation, limiting cases, rank roles, and theory-to-implementation consequences.

## Core notation

Mathematical statements use uppercase $X$ and $Y$ for predictor and response matrices. Python
estimator signatures retain scikit-learn's `X` and `y` convention, including when `y` is a
two-dimensional multivariate response matrix. The programming name does not change the
mathematical object.

Complete matrices are bold. Use `\mathbf` for Latin matrix symbols and `\boldsymbol` for Greek
matrix symbols that must render in bold. Descriptive, role, block, method, and extremum subscripts
are upright with `\mathrm`, for example $d_{\mathrm{p}}$,
$\boldsymbol{\Lambda}_{\mathrm{s}}$, $\mathbf{L}_{\mathrm{sp}}$,
$\mathbf{U}_{\mathrm{X}}$, and $h_{\mathrm{max}}$. Mathematical indices and dimensions remain
italic, for example $D_k$, $D_{kk}$, $P_{:k}$, $s_i$, $r_\pi$, and $\mathbf{I}_p$.

| Quantity | Shape | Meaning |
|---|---:|---|
| $\mathbf{X}_{\mathrm{cs}}$ | $(n,p)$ | centered, optionally scaled predictors |
| $\mathbf{Y}_{\mathrm{cs}}$ | $(n,q)$ | centered, optionally scaled responses |
| $\mathbf{\Pi}$ | $(p,r_\pi)$ | retained predictor basis from the leading right singular vectors of $\mathbf{X}_{\mathrm{cs}}$ |
| $\mathbf{Z}$ | $(n,r_\pi)$ | retained predictor scores, $\mathbf{Z}=\mathbf{X}_{\mathrm{cs}}\mathbf{\Pi}$ |
| $\mathbf{C}$ | $(q,h)$ | orthonormal response basis selected by the configured response-subspace policy |
| $\mathbf{W}$ | $(r_\pi,h)$ | least-squares map from $\mathbf{Z}$ to $\mathbf{Y}_{\mathrm{cs}}\mathbf{C}$ |
| $\mathbf{M}$ | $(r_\pi,h)$ | left singular vectors of $\mathbf{W}$ |
| $\mathbf{D}$ | $(h,h)$ | nonnegative diagonal dilation matrix |
| $\mathbf{N}$ | $(h,h)$ | right singular vectors of $\mathbf{W}$ |
| $\mathbf{P}$ | $(p,h)$ | orthonormal predictor directions, $\mathbf{P}=\mathbf{\Pi}\mathbf{M}$ |
| $\mathbf{Q}$ | $(q,h)$ | orthonormal response directions, $\mathbf{Q}=\mathbf{C}\mathbf{N}$ |

Admissibility requires

\begin{equation}
1\le h\le\min(r_\pi,q),
\qquad
r_\pi\le\min(n,p).
\end{equation}

## Terminology and orientation

$\mathbf{\Pi}$ is the retained predictor basis and
$\mathbf{\Pi}\mathbf{\Pi}^{\mathsf T}$ is the retained-subspace projector. The columns of
$\mathbf{P}$ and $\mathbf{Q}$ are orthonormal predictor and response directions. The projectors
onto their final spans are $\mathbf{P}\mathbf{P}^{\mathsf T}$ and
$\mathbf{Q}\mathbf{Q}^{\mathsf T}$; neither $\mathbf{P}$ nor $\mathbf{Q}$ is itself a
projection matrix.

For each $k$, $D_k=D_{kk}$ is the dilation of paired latent mode
$(P_{:k},D_k,Q_{:k})$. The matrices $\mathbf{X}_{\mathrm{cs}}\mathbf{P}$ and
$\mathbf{Y}_{\mathrm{cs}}\mathbf{Q}$ contain predictor and response scores. Public
`n_components` counts the $h$ paired latent modes, while public `predictor_rank` denotes the retained
predictor-subspace dimension $r_\pi$.

The package's weighted response directions have response-by-mode orientation
$\mathbf{Q}\mathbf{D}$, and

\begin{equation}
\mathbf{D}\mathbf{Q}^{\mathsf T}
=
(\mathbf{Q}\mathbf{D})^{\mathsf T}.
\end{equation}

The public decomposition fields are `predictor_directions` and `response_directions`. The standard
PLS-style fitted attributes remain `x_rotations_` and `y_rotations_`. Predictor and response
directions are not the estimator's separate least-squares reconstruction loadings.

The private core additionally requires $r_\pi$ not to exceed the numerical rank of the supplied
predictor matrix. For thin-SVD singular values $s_i$, numerical rank is determined by
$s_i>\tau_{\mathrm{X}}$, where

\begin{equation}
\tau_{\mathrm{X}}=\max(n,p)\,\epsilon_{64}\,s_1.
\end{equation}

## Fixed-parameter construction

For the thin SVD

\begin{equation}
\mathbf{X}_{\mathrm{cs}}
=\mathbf{U}_{\mathrm{X}}\mathbf{S}_{\mathrm{X}}\mathbf{V}_{\mathrm{X}}^{\mathsf T},
\end{equation}

retain the first $r_\pi$ right singular vectors in $\mathbf{\Pi}$. The retained-subspace projector
and predictor decomposition are

\begin{equation}
\mathbf{\Pi}^{\mathsf T}\mathbf{\Pi}=\mathbf{I}_{r_\pi},
\end{equation}

\begin{equation}
\mathbf{X}_{\mathrm{cs}}
=
\mathbf{X}_{\mathrm{cs}}\mathbf{\Pi}\mathbf{\Pi}^{\mathsf T}
+
\mathbf{X}_{\mathrm{cs}}
(\mathbf{I}_p-\mathbf{\Pi}\mathbf{\Pi}^{\mathsf T}).
\end{equation}

Set

\begin{equation}
\mathbf{Z}=\mathbf{X}_{\mathrm{cs}}\mathbf{\Pi}.
\end{equation}

The configured `response_subspace` policy selects an orthonormal response basis satisfying

\begin{equation}
\mathbf{C}^{\mathsf T}\mathbf{C}=\mathbf{I}_h.
\end{equation}

For the default peer-reviewed policy, `"cross_covariance"`, the response basis solves

\begin{equation}
\max_{\mathbf{C}^{\mathsf T}\mathbf{C}=\mathbf{I}_h}
\left\|
\mathbf{Z}^{\mathsf T}\mathbf{Y}_{\mathrm{cs}}\mathbf{C}
\right\|_{\mathrm{F}}^2.
\end{equation}

One optimum is formed by the leading $h$ right singular vectors of
$\mathbf{Z}^{\mathsf T}\mathbf{Y}_{\mathrm{cs}}$.

For the software-only `"least_squares"` policy, which is not part of the peer-reviewed companion
publication, choose $\mathbf{C}$ and $\mathbf{W}$ jointly by

\begin{equation}
(\mathbf{C}_{\mathrm{LS}},\mathbf{W}_{\mathrm{LS}})
=
\arg\min_{\substack{\mathbf{C}^{\mathsf T}\mathbf{C}=\mathbf{I}_h\\
\mathbf{W}\in\mathbb{R}^{r_\pi\times h}}}
\left\|
\mathbf{Y}_{\mathrm{cs}}-\mathbf{Z}\mathbf{W}\mathbf{C}^{\mathsf T}
\right\|_{\mathrm{F}}^2.
\end{equation}

With $\mathbf{P}_{\mathbf{Z}}=\mathbf{Z}\mathbf{Z}^{+}$, eliminating $\mathbf{W}$ gives

\begin{equation}
\mathbf{C}_{\mathrm{LS}}
=
\arg\max_{\mathbf{C}^{\mathsf T}\mathbf{C}=\mathbf{I}_h}
\operatorname{tr}\left[
\mathbf{C}^{\mathsf T}\mathbf{Y}_{\mathrm{cs}}^{\mathsf T}
\mathbf{P}_{\mathbf{Z}}\mathbf{Y}_{\mathrm{cs}}\mathbf{C}
\right].
\end{equation}

This is the rank-$h$ reduced-rank-regression response subspace for regression of
$\mathbf{Y}_{\mathrm{cs}}$ on the retained predictor coordinates $\mathbf{Z}$. For fixed
$(h,r_\pi)$, its training residual cannot exceed that of the cross-covariance policy, apart from
numerical tolerance. The two policies have the same fitted regression map when $q=1$ and when the
complete response space is retained with $h=q\le r_\pi$.

Solve

\begin{equation}
\mathbf{Z}\mathbf{W}
\approx
\mathbf{Y}_{\mathrm{cs}}\mathbf{C}
\end{equation}

by minimum-norm least squares, then factor

\begin{equation}
\mathbf{W}=\mathbf{M}\mathbf{D}\mathbf{N}^{\mathsf T},
\qquad
\mathbf{D}=\operatorname{diag}(D_1,\ldots,D_h),
\qquad
D_1\ge\cdots\ge D_h\ge0.
\end{equation}

Define

\begin{equation}
\mathbf{P}=\mathbf{\Pi}\mathbf{M},
\qquad
\mathbf{Q}=\mathbf{C}\mathbf{N}.
\end{equation}

The resulting invariants are

\begin{equation}
\mathbf{P}^{\mathsf T}\mathbf{P}=\mathbf{I}_h,
\qquad
\mathbf{Q}^{\mathsf T}\mathbf{Q}=\mathbf{I}_h,
\end{equation}

and the diagonal latent relation is

\begin{equation}
\mathbf{Y}_{\mathrm{cs}}\mathbf{Q}
=
\mathbf{X}_{\mathrm{cs}}\mathbf{P}\mathbf{D}
+
\mathbf{E}_\pi.
\end{equation}

The centered/scaled regression map is

\begin{equation}
\mathbf{B}_{\mathrm{cs}}
=
\mathbf{P}\mathbf{D}\mathbf{Q}^{\mathsf T}
=
\mathbf{\Pi}\mathbf{W}\mathbf{C}^{\mathsf T}.
\end{equation}

Fitted values must agree under both factorizations:

\begin{equation}
\mathbf{X}_{\mathrm{cs}}\mathbf{B}_{\mathrm{cs}}
=
\mathbf{Z}\mathbf{W}\mathbf{C}^{\mathsf T}.
\end{equation}

After $\mathbf{\Pi}$ is fixed, the nominal fitted dimension is

\begin{equation}
(r_\pi+q-h)h.
\end{equation}

This count is based on $\mathbf{M}$, $\mathbf{D}$, and $\mathbf{Q}$; the entries of
$\mathbf{P}=\mathbf{\Pi}\mathbf{M}$ are not independently free.

Repeated or nearly repeated singular values identify invariant subspaces, not intrinsically
numbered basis vectors. Tests must compare projection matrices, principal angles, singular values,
regression maps, or predictions rather than raw basis columns.

## Path-selection boundary

For path selection, let $r_{\mathrm{num,min}}$ be the minimum predictor rank verified after
fold-local pipeline preprocessing and terminal-estimator centering/scaling. The hard/default path
ceiling is

\begin{equation}
r_{\pi,\mathrm{hard}}
=
\min\left[
 p_{\mathrm{min}},
 n_{\mathrm{train,min}}-1,
 r_{\mathrm{num,min}}
\right].
\end{equation}

With `max_predictor_rank=None`, automatic search uses this complete hard-feasible domain. For an
explicit integer $r_{\mathrm{user}}$, the effective search ceiling is

\begin{equation}
r_{\pi,\mathrm{max}}
=
\min\left[r_{\pi,\mathrm{hard}},r_{\mathrm{user}}\right].
\end{equation}

Otherwise $r_{\pi,\mathrm{max}}=r_{\pi,\mathrm{hard}}$.

The EPV-inspired policy is separate. For `predictor_rank_values="epv"`, its nominal full-sample
rank is

\begin{equation}
r_{\pi,\mathrm{epv}}
=
\min\left[p,\left\lceil\frac{n}{c}\right\rceil\right],
\end{equation}

and the evaluated fixed rank is clipped only by $r_{\pi,\mathrm{max}}$. The samples-per-rank
quantity $c$ does not restrict ordinary automatic or exhaustive search.

These search policies are package contracts around the fixed model rather than part of the
mathematical definition above.
