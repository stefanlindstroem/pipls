# Π-PLS theory overview

Π-PLS is a multivariate linear-regression method with two explicit rank controls. It first retains
a rank-controlled predictor subspace, then constructs a response subspace under a configured
selection criterion, and finally diagonalizes the reduced regression map into paired
predictor-response modes. The package default uses the cross-covariance construction from the
[peer-reviewed companion publication](citation.md#companion-paper).

## Scientific source and package scope

The scientific source for the construction summarized on this page is the
[peer-reviewed companion publication](citation.md#companion-paper):

> Vishal Agrawal, Fritjof Nilsson, and Stefan B. Lindström, “Panoramic Partial Least Squares
> (Π-PLS): A transparent and parsimonious multivariate regression model with paired latent
> directions.” *Computers & Chemical Engineering*, 109913 (2026).
> [https://doi.org/10.1016/j.compchemeng.2026.109913](https://doi.org/10.1016/j.compchemeng.2026.109913)

The manuscript states the fixed mathematical core for centered predictor and response matrices,
including the cross-covariance response-subspace construction. The package implements that
peer-reviewed construction as its default and adds ordinary software facilities around it, including
optional scaling, numerical-rank checks, configurable predictor SVD solvers, cross-validated search,
immutable result records, and prediction diagnostics. The package also provides one
least-squares/RRR-inspired response-subspace construction for programming users, which can show
Pareto dominance for some datasets. That alternative is a software extension and is not part of the
[peer-reviewed companion publication](citation.md#companion-paper).

Reference dataset loaders are provided for demonstration purposes. Provenance and analysis are
covered in the [reference-dataset guide](datasets.md).

## Problem setting and two rank controls {#problem-setting-and-two-rank-controls}

Let $\mathbf{X}\in\mathbb{R}^{n\times p}$ contain column-centered predictors and let
$\mathbf{Y}\in\mathbb{R}^{n\times q}$ contain column-centered responses. Multivariate linear
regression seeks
$\mathbf{B}\in\mathbb{R}^{p\times q}$ such that

\begin{equation}
\mathbf{Y}=\mathbf{X}\mathbf{B}+\mathbf{E}.
\end{equation}

Π-PLS separates two structural choices:

- the predictor rank $r_\pi$, which controls the dimension of the retained predictor subspace;
- the component count $h$, which controls the number of paired predictor-response modes.

The admissible dimensions satisfy

\begin{equation}
1\leq h\leq\min(r_\pi,q),
\qquad
r_\pi\leq\min(n,p).
\end{equation}

For a centered estimator fit, the effective sample-space bound becomes
$r_\pi\leq\min(n-1,p)$. The implementation also requires $r_\pi$ not to exceed the verified numerical
rank of the preprocessed predictor matrix.

## Canonical terminology {#canonical-terminology}

The package uses the following terms for the fixed Π-PLS construction:

| Object or public name | Canonical meaning |
|---|---|
| $\mathbf{\Pi}$ | retained predictor basis |
| $\mathbf{\Pi}\mathbf{\Pi}^{\mathsf T}$ | retained-subspace projector |
| $\mathbf{P}$ | orthonormal predictor directions |
| $\mathbf{Q}$ | orthonormal response directions |
| $D_k=D_{kk}$ | dilation of paired latent mode $k$ |
| $\mathbf{X}\mathbf{P}$ | predictor scores |
| $\mathbf{Y}\mathbf{Q}$ | response scores |
| $(\mathbf{P}_{:k},D_k,\mathbf{Q}_{:k})$ | paired latent mode $k$ |
| `predictor_rank` | retained predictor-subspace dimension $r_\pi$ |
| `n_components` | number of paired latent modes $h$ |

The word *direction* is the mathematical term for columns of $\mathbf{P}$ and
$\mathbf{Q}$. `PiPLSDecomposition` exposes these arrays as `predictor_directions` and
`response_directions`. The estimator retains the standard PLS-style names `x_rotations_` and
`y_rotations_`.

The Π-PLS directions are also distinct from `x_loadings_` and `y_loadings_`, which are
least-squares reconstruction loadings for the centered or centered-and-scaled training blocks.
For response-side factor displays, the package stores response-by-mode weighted directions
$\mathbf{Q}\mathbf{D}$, where column $k$ is $D_kQ_{:k}$.

The API word “component” is retained because it is familiar in regression software. In Π-PLS,
`n_components` counts paired latent modes. Likewise, `predictor_rank` is a retained
observed-subspace dimension; the term *predictor signal rank* is reserved for synthetic
settings where the noiseless generating rank is known.

## 1. Rank-controlled predictor projection {#rank-controlled-predictor-projection}

Take a singular value decomposition

\begin{equation}
\mathbf{X}=\mathbf{U}_{\mathrm{X}}\mathbf{S}_{\mathrm{X}}\mathbf{V}_{\mathrm{X}}^{\mathsf T}
\end{equation}

and retain the leading $r_\pi$ right singular vectors:

\begin{equation}
\mathbf{\Pi}=\mathbf{V}_{\mathrm{X}}(:,1:r_\pi)\in\mathbb{R}^{p\times r_\pi},
\qquad
\mathbf{\Pi}^{\mathsf T}\mathbf{\Pi}=\mathbf{I}_{r_\pi}.
\end{equation}

The matrix $\mathbf{\Pi}\mathbf{\Pi}^{\mathsf T}$ is the orthogonal projector onto the retained
predictor subspace, so

\begin{equation}
\mathbf{X}=\mathbf{X}\mathbf{\Pi}\mathbf{\Pi}^{\mathsf T}+\mathbf{X}(\mathbf{I}_p-\mathbf{\Pi}\mathbf{\Pi}^{\mathsf T}).
\end{equation}

The first term, $\mathbf{X}\mathbf{\Pi}\mathbf{\Pi}^{\mathsf T}$, is the rank-$r_\pi$ predictor
approximation. The second term is orthogonal to the retained subspace and contributes a
truncation residual. Substituting this decomposition into the regression relation gives

\begin{equation}
\mathbf{Y}=\mathbf{Z}\mathbf{\Pi}^{\mathsf T}\mathbf{B}+\mathbf{E}',
\qquad
\mathbf{E}'=\mathbf{E}+\mathbf{X}(\mathbf{I}_p-\mathbf{\Pi}\mathbf{\Pi}^{\mathsf T})\mathbf{B},
\end{equation}

where $\mathbf{Z}=\mathbf{X}\mathbf{\Pi}\in\mathbb{R}^{n\times r_\pi}$ is the retained predictor scores.

## 2. Response-subspace selection {#response-subspace-selection}

After fixing the retained predictor representation $\mathbf{Z}=\mathbf{X}\mathbf{\Pi}$, the
package supports two criteria for selecting an $h$-dimensional orthonormal response basis
$\mathbf{C}\in\mathbb{R}^{q\times h}$. Both choices feed the same later least-squares coupling
and diagonalization. They differ only in how the intermediate response subspace is chosen.

### 2.1 Cross-covariance selection

The [peer-reviewed companion publication](citation.md#companion-paper) uses the cross-covariance
construction, which is also the package default (`response_subspace="cross_covariance"`). Define

\begin{equation}
\boldsymbol{\Sigma}_{\mathrm{ZY}}=\mathbf{Z}^{\mathsf T}\mathbf{Y}\in\mathbb{R}^{r_\pi\times q}.
\end{equation}

For prescribed $h$, the response basis is chosen by

\begin{equation}
\mathbf{C}_{\mathrm{cov}}
=
\arg\max_{\mathbf{C}^{\mathsf T}\mathbf{C}=\mathbf{I}_h}
\left\|\boldsymbol{\Sigma}_{\mathrm{ZY}}\mathbf{C}\right\|_{\mathrm{F}}^2,
\end{equation}

and its solution is obtained through an SVD as

\begin{equation}
\mathbf{C}_{\mathrm{cov}}=\mathbf{V}_{\mathrm{cov}}(:,1:h),\qquad \boldsymbol{\Sigma}_{\mathrm{ZY}}=\mathbf{U}_{\mathrm{cov}}\mathbf{S}_{\mathrm{cov}}\mathbf{V}_{\mathrm{cov}}^{\mathsf T}.
\end{equation}

Thus the default policy selects the response subspace with the largest retained squared
cross-covariance with $\mathbf{Z}$. The maximizing subspace is the meaningful object; individual
basis vectors are not unique when relevant singular values are repeated.

### 2.2 Least-squares selection

The alternative `response_subspace="least_squares"` chooses the response subspace jointly with a
rank-$h$ reduced regression map by solving

\begin{equation}
(\mathbf{C}_{\mathrm{LS}},\mathbf{W}_{\mathrm{LS}})
=
\arg\min_{\substack{\mathbf{C}^{\mathsf T}\mathbf{C}=\mathbf{I}_h\\
\mathbf{W}\in\mathbb{R}^{r_\pi\times h}}}
\left\|\mathbf{Y}-\mathbf{Z}\mathbf{W}\mathbf{C}^{\mathsf T}\right\|_{\mathrm{F}}^2.
\end{equation}

For fixed $\mathbf{C}$, the minimizing map is

\begin{equation}
\mathbf{W}=\mathbf{Z}^{+}\mathbf{Y}\mathbf{C}.
\end{equation}

Let $\mathbf{P}_{\mathbf{Z}}=\mathbf{Z}\mathbf{Z}^{+}$ be the orthogonal projector onto the retained
predictor-score column space. Eliminating $\mathbf{W}$ gives the equivalent response-subspace problem

\begin{equation}
\mathbf{C}_{\mathrm{LS}}
=
\arg\max_{\mathbf{C}^{\mathsf T}\mathbf{C}=\mathbf{I}_h}
\operatorname{tr}\!\left(
\mathbf{C}^{\mathsf T}
\mathbf{Y}^{\mathsf T}
\mathbf{P}_{\mathbf{Z}}
\mathbf{Y}
\mathbf{C}
\right).
\end{equation}

This is the rank-$h$ reduced-rank-regression response subspace for regression of $\mathbf{Y}$ on
the fixed retained predictor coordinates $\mathbf{Z}$. Consequently, for the same fixed
$(h,r_\pi)$, the least-squares policy minimizes the training Frobenius residual over admissible
rank-$h$ maps.

With a reduced QR factorization

\begin{equation}
\mathbf{Z}=\mathbf{Q}_{\mathbf{Z}}\mathbf{R}_{\mathbf{Z}},
\qquad
\mathbf{Q}_{\mathbf{Z}}^{\mathsf T}\mathbf{Q}_{\mathbf{Z}}=\mathbf{I}_{r_\pi},
\end{equation}

we have $\mathbf{P}_{\mathbf{Z}} = \mathbf{Q}_{\mathbf{Z}}\mathbf{Q}_{\mathbf{Z}}^{\mathsf T}$, and therefore

\begin{equation}
\mathbf{Y}^{\mathsf T}\mathbf{P}_{\mathbf{Z}}\mathbf{Y}
=
(\mathbf{Q}_{\mathbf{Z}}^{\mathsf T}\mathbf{Y})^{\mathsf T}
(\mathbf{Q}_{\mathbf{Z}}^{\mathsf T}\mathbf{Y}),
\end{equation}

The columns of $\mathbf{C}_{\mathrm{LS}}$ are thus the leading right singular directions of
$\mathbf{Q}_{\mathbf{Z}}^{\mathsf T}\mathbf{Y}$.

> **Software-extension boundary.** The least-squares response-subspace construction is implemented
> for programming users but is not part of the
> [peer-reviewed companion publication](citation.md#companion-paper).
> Manuscript-aligned analyses use `response_subspace="cross_covariance"`.


## 3. Least-squares coupling in the selected response subspace

Under either response-subspace policy, project the response matrix onto the selected basis and fit

\begin{equation}
\mathbf{Y}\mathbf{C}=\mathbf{Z}\mathbf{W}+\mathbf{E}'',
\end{equation}

where $\mathbf{W}\in\mathbb{R}^{r_\pi\times h}$. The minimum-norm least-squares solution is

\begin{equation}
\mathbf{W}=\mathbf{Z}^{+}\mathbf{Y}\mathbf{C}.
\end{equation}

Before diagonalization, the regression map in the centered coordinates is

\begin{equation}
\mathbf{B}_{\mathrm{cs}}=\mathbf{\Pi}\mathbf{W}\mathbf{C}^{\mathsf T},
\end{equation}

and the fitted responses are

\begin{equation}
\widehat{\mathbf{Y}}=\mathbf{X}\mathbf{B}_{\mathrm{cs}}=\mathbf{Z}\mathbf{W}\mathbf{C}^{\mathsf T}.
\end{equation}

The implementation uses a least-squares solver rather than forming an explicit inverse.

## 4. Diagonal latent coupling {#diagonal-latent-coupling}

Take the economy-size singular value decomposition

\begin{equation}
\mathbf{W}=\mathbf{M}\mathbf{D}\mathbf{N}^{\mathsf T},
\end{equation}

where $\mathbf{M}\in\mathbb{R}^{r_\pi\times h}$ and
$\mathbf{N}\in\mathbb{R}^{h\times h}$ have orthonormal columns,
and

\begin{equation}
\mathbf{D}=\operatorname{diag}(D_1,\ldots,D_h),
\qquad
D_1\geq D_2\geq\cdots\geq D_h\geq0.
\end{equation}

Define

\begin{equation}
\mathbf{P}=\mathbf{\Pi}\mathbf{M}\in\mathbb{R}^{p\times h},
\qquad
\mathbf{Q}=\mathbf{C}\mathbf{N}\in\mathbb{R}^{q\times h},
\end{equation}

where $\mathbf{P}^{\mathsf T}\mathbf{P}=\mathbf{I}_h$ and $\mathbf{Q}^{\mathsf T}\mathbf{Q}=\mathbf{I}_h$,
and the reduced regression relation becomes

\begin{equation}
\mathbf{Y}\mathbf{Q}=\mathbf{X}\mathbf{P}\mathbf{D}+\mathbf{E}_\pi,
\qquad
\mathbf{E}_\pi=\mathbf{E}''\mathbf{N}.
\end{equation}

The predictor score vector $\mathbf{X}\mathbf{P}_{:k}$ is coupled only to the response score
vector $\mathbf{Y}\mathbf{Q}_{:k}$, with dilation $D_k$. This is the one-to-one, mode-wise interpretation
central to Π-PLS.

The equivalent regression-map factorizations are

\begin{equation}
\boxed{
\mathbf{B}_{\mathrm{cs}}
=\mathbf{P}\mathbf{D}\mathbf{Q}^{\mathsf T}
=\mathbf{\Pi}\mathbf{W}\mathbf{C}^{\mathsf T}
}.
\end{equation}

For new centered predictors $\mathbf{X}_{\mathrm{new}}$,

\begin{equation}
\widehat{\mathbf{Y}}_{\mathrm{new}}=\mathbf{X}_{\mathrm{new}}\mathbf{P}\mathbf{D}\mathbf{Q}^{\mathsf T}.
\end{equation}

## Why the method is called panoramic {#why-the-method-is-panoramic}

Standard deflation-based PLS algorithms construct successive components while removing previously
modelled predictor variation. Π-PLS instead fixes one rank-controlled predictor representation
$\mathbf{Z}=\mathbf{X}\mathbf{\Pi}$ and derives all $h$ coupled modes from that undeflated retained
space through closed-form matrix decompositions. The retained predictor space therefore remains
available as a whole during response-subspace selection and regression; in this sense, the view is
panoramic.

## Ranks and fitted dimension {#interpretation-of-the-ranks}

The two ranks control different forms of complexity.

The predictor rank $r_\pi$ determines which predictor variation remains available to the model. If
$r_\pi$ is too small, response-relevant variation can be discarded irreversibly. Retaining more
predictor directions does not force all of them to contribute strongly, because the response
subspace and latent least-squares map are estimated afterward.

The component count $h$ determines the rank and interpretive complexity of the diagonal
predictor-response coupling. Increasing $h$ adds paired modes, but it cannot recover predictor
directions excluded by $r_\pi$ and cannot exceed the number of responses.

After $\mathbf{\Pi}$ has been fixed, the fitted representation is described by $\mathbf{M}$,
$\mathbf{D}$, and $\mathbf{Q}$. These contain $r_\pi h$, $h$, and $qh$ entries. The orthonormality
constraints on $\mathbf{M}$ and $\mathbf{Q}$ each remove $h(h+1)/2$ degrees of freedom. The resulting
nominal fitted dimension is

\begin{equation}
\boxed{(r_\pi+q-h)h}.
\end{equation}


## Relationships to established methods {#relationships-to-established-methods}

### Canonical correlation analysis {#canonical-correlation-analysis}

Like Π-PLS, CCA also constructs paired predictor and response variates with a diagonal
association structure, but classical CCA maximizes normalized correlation after within-block whitening. The default
Π-PLS response policy instead uses an unwhitened cross-covariance criterion inside the retained
predictor representation, while the optional least-squares policy uses the fitted-response
least-squares criterion described above. Both policies then form the same diagonal paired-mode
representation. The structural analogy to CCA therefore concerns the final paired relation, not an
identity of objectives.

### Reduced-rank regression {#reduced-rank-regression}

Both Π-PLS and reduced-rank regression produce low-rank coefficient structures. Reduced-rank
regression obtains the best rank-$h$ approximation of the OLS fit under its least-squares
criterion. With `response_subspace="least_squares"`, Π-PLS applies exactly that reduced-rank
principle after first fixing the retained predictor representation $\mathbf{Z}$, and then
diagonalizes the resulting reduced map into one-to-one paired modes. With the default
`"cross_covariance"` policy, the response subspace is instead selected by the
[peer-reviewed cross-covariance criterion](citation.md#companion-paper) before the common
least-squares coupling and diagonalization stages.

### PLS and PLS-SVD {#pls-and-pls-svd}

Standard multicomponent PLS is commonly constructed through iterative covariance-guided extraction
and deflation. Π-PLS instead fixes one retained predictor space and then selects a response
subspace before fitting and diagonalizing the reduced least-squares map. Under the publication
default, response-subspace selection is itself cross-covariance-driven; under the optional
least-squares policy, it is RRR-inspired.

PLS-SVD derives predictor and response directions directly from a cross-covariance operator. Under
the default Π-PLS policy, $\mathbf{Z}^{\mathsf T}\mathbf{Y}$ selects only the intermediate response
subspace after $\mathbf{Z}=\mathbf{X}\mathbf{\Pi}$ has already been determined from predictor
singular structure. Under the least-squares policy, that intermediate subspace is instead selected
from the predictable response variation in $\operatorname{col}(\mathbf{Z})$. In both cases, the
final $\mathbf{P}$ and $\mathbf{Q}$ arise only after the common least-squares map $\mathbf{W}$ is
estimated and diagonalized.

## Selection, validation, and synthetic-data boundaries {#selection-validation-and-synthetic-data-boundaries}

The equations above define one fixed $(h,r_\pi)$ model under one configured response-subspace
policy. `PiPLSSearchCV` is a package-level facility for evaluating admissible fixed models under
fold-local preprocessing and response-standardized CV-MSE. It searches component count and
predictor rank; it does not automatically compare response-subspace policies. A controlled
programming-user comparison therefore uses two estimator templates with the same materialized CV
splits. Its search policies and the practical real-data workflows documented elsewhere are not
redefined by this theory page and need not duplicate the manuscript’s experimental choices.

`make_synthetic_data()` provides the package's deterministic synthetic-data utility and implements
the [companion publication’s](citation.md#companion-paper) Gaussian latent data-generating model
directly. See [Datasets and generators](api/datasets.md#synthetic-generator) for the exact
distribution and generator contract, and [Reference datasets](datasets.md) for packaged data.

The [Pulp tutorial’s conditional-rank selection section](tutorials/pulp.md#retrieve-selection-evidence)
and [Path and selection](path_selection.md) define the package’s general path-construction,
validation, and selection contracts.
