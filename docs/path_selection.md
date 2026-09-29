# Path and selection

A `PiPLSRegression` fit uses one pair $(h,r_\pi)$: paired-mode count $h$ (`n_components`) and
retained predictor-subspace dimension $r_\pi$ (`predictor_rank`). Their distinct mathematical roles
are summarized under [Problem setting and two rank controls](theory.md#problem-setting-and-two-rank-controls).
`PiPLSSearchCV` turns candidate pairs into a component path from which one model can be selected.

The workflow is: choose how $r_\pi$ is handled, scan $h$, inspect the component path, select $h$,
and refit the retained pair. Predictor rank can be **conditional**, with one $r_\pi$ selected at
each $h$, or **fixed** before the component path is evaluated. The sections below first define the
feasible pairs, then describe conditional and fixed-rank searches, component selection, and
refitting. For constructor signatures and fitted attributes, see
[`PiPLSSearchCV`](api/path.md).

## Search domain { #search-bounds }

During cross-validation, each candidate pair $(h,r_\pi)$ must be feasible in every training split
used to evaluate it. The search domain is therefore the intersection of the splitwise feasible
domains.

For training split $j$, let $n_j$ be its number of training observations, let $p_j$ be the
predictor count presented to the terminal `PiPLSRegression` after fold-local preprocessing, and let
$r_j$ be its verified numerical predictor rank after terminal-estimator centering and optional
scaling. Define the corresponding minima across all training splits as

\begin{equation}
n'=\min_j n_j,
\qquad
p'=\min_j p_j,
\qquad
r'=\min_j r_j.
\end{equation}

Cross-validation itself changes rows, not predictor columns. The predictor count $p_j$ can vary only
when a supported scikit-learn `Pipeline` contains fold-fitted preprocessing that changes predictor
dimensionality, such as feature selection. For a direct `PiPLSRegression` fit, or for
dimension-preserving preprocessing, we have $p'=p$.

A predictor rank used by the search must be feasible in every split. The hard predictor-rank ceiling
is therefore

\begin{equation}
r_{\pi,\mathrm{hard}}
=
\min(p',n'-1,r').
\end{equation}

The term $n'-1$ reflects the loss of one predictor dimension under centering. The verified rank
$r'$ also captures any additional numerical rank loss after preprocessing, centering, and optional
scaling.

With `max_predictor_rank=None`, the search uses this hard ceiling $r_{\pi,\mathrm{max}} = r_{\pi,\mathrm{hard}}$. An explicit positive integer
`max_predictor_rank=k` adds a user restriction

\begin{equation}
r_{\pi,\mathrm{max}}
=
\min(r_{\pi,\mathrm{hard}},k).
\end{equation}

Let $q$ be the number of responses, which is common to all training splits. For the ordinary full
rank domain, the admissible integer pairs are

\begin{equation}
\mathcal{D}
=
\{(h,r_\pi)\in\mathbb{N}^2:1\le h\le \min(q,r_{\pi,\mathrm{max}}),\ h\le r_\pi\le r_{\pi,\mathrm{max}}\}.
\end{equation}

The domain $\mathcal{D}$ is triangular: increasing $h$ removes all predictor ranks below $h$.
`predictor_rank_values=None` uses every feasible integer $r_\pi$ in this domain, and
`n_components_values="all"` uses every feasible integer $h$. Explicit component or predictor-rank
values restrict this domain after the same feasibility checks.

The Pulp dataset gives a compact example. Its exhaustive search has $q=8$ and
$r_{\pi,\mathrm{max}}=14$, giving 84 admissible pairs. Each colored cell below is one pair
evaluated under the maintained protocol of ten repetitions of five-fold CV; color gives its actual
mean response-standardized CV-MSE. Gray cells violate $r_\pi\ge h$.

![Pulp exhaustive search domain](assets/generated/pulp/search_domain.svg)

These bounds describe feasibility. Selection begins only after the admissible pairs have been
defined.

## Exhaustive search and selection { #exhaustive-search-and-selection }

With `predictor_rank_values=None` and `search_method="exhaustive"`, every pair in $\mathcal{D}$
is evaluated. Under the default scorer, each cell in the Pulp figure therefore contains one mean
response-standardized CV-MSE, denoted $M_{h,r_\pi}$ below. Lower values are better.

### Conditional predictor-rank selection

Predictor rank is selected separately at each component count. For fixed $h$, let

\begin{equation}
M_{h,\min}=\min_{r_\pi:(h,r_\pi)\in\mathcal{D}} M_{h,r_\pi}.
\end{equation}

A predictor rank qualifies when its mean CV-MSE satisfies both

\begin{equation}
M_{h,r_\pi} \le (1+\delta_{\pi,\mathrm{rel}})M_{h,\min},
\qquad
M_{h,r_\pi} \le M_{h,\min}+\delta_{\pi,\mathrm{abs}}.
\end{equation}

The smallest qualifying $r_\pi$ is retained. Thus the tolerance can trade a small increase in
validation error for a smaller predictor subspace without changing which candidates were evaluated.
`predictor_rank_relative_tolerance=None` resolves to `sqrt(np.finfo(np.float64).eps)`, while an
infinite `predictor_rank_absolute_tolerance` disables the absolute cap.

The next figure applies the same rule to the already evaluated Pulp surface. To make the mechanism
visible, it uses a deliberately generous 15% relative predictor-rank tolerance and no absolute cap.
Open circles mark the exact minimum at each $h$; diamonds mark the smallest qualifying rank. This
illustrative tolerance does not require another exhaustive CV run because candidate evaluation is
independent of the tolerance.

![Pulp conditional predictor-rank selection](assets/generated/pulp/conditioned_search_domain.svg)

Write the retained rank as $\hat r_\pi(h)$. The two-dimensional search has now been reduced to one
retained pair $(h,\hat r_\pi(h))$ for each component count. 

### Component path and component selection

The retained pairs $(h,\hat r_\pi(h))$ form the *component path*. Define its mean CV-MSE at component count $h$ as
$M_h=M_{h,\hat r_\pi(h)}$.

These CV-MSE values $M_h$ are used for manual selection of the number of components $h$ in,
*e.g.*, the [Complete Pulp Analysis](tutorials/pulp.md) tutorial.

For automated component selection, let

\begin{equation}
M_{\min}=\min_h M_h.
\end{equation}

For `rule="minimum_cv_mse"`, a component count qualifies when

\begin{equation}
M_h \le (1+\delta_{\mathrm{rel}})M_{\min},
\qquad
M_h \le M_{\min}+\delta_{\mathrm{abs}}.
\end{equation}

Because the path is ordered by increasing $h$, the first qualifying row is retained.
`relative_tolerance=None` again resolves to `sqrt(np.finfo(np.float64).eps)`, and an infinite
`absolute_tolerance` disables the absolute cap.

For visibility, the Pulp component-path figure below continues the 15% predictor-rank illustration
and then applies a deliberately generous 50% relative component tolerance. The exact path minimum
is at $h=3$, while the smaller $h=2$ model falls inside that illustrative tolerance and is retained.
The labels above the path show the predictor rank already selected conditionally at each $h$.

![Pulp conditioned component-path selection](assets/generated/pulp/conditioned_component_path.svg)

This ordering is central: predictor rank is resolved first at each $h$; component selection then
acts on the resulting one-dimensional path and does not revisit the predictor-rank profiles.

## Fixed predictor rank: EPV { #epv-policy }

The exhaustive case above optimizes predictor rank separately at every component count, which is computationally expensive. The EPV
policy instead fixes one predictor rank before the component path is evaluated. With
`predictor_rank_values="epv"`, the rank is

\begin{equation}
r_{\pi,\mathrm{epv}}
=
\min\left(p,\left\lceil\frac{n}{c}\right\rceil,r_{\pi,\mathrm{max}}\right),
\end{equation}

where $n$ and $p$ are the full-data observation and predictor counts supplied to `fit()` and $c$ is
`samples_per_predictor_rank`. This is also the fixed-rank workflow used for the real-data analyses in
the [companion paper](citation.md#companion-paper): $r_\pi$ is set by the EPV-inspired rule before
$h$ is selected by cross-validation.

The default is $c=10$. However, the Pulp dataset has only $n=46$ observations. For this
illustration, the more permissive $c=5$ is used by setting `samples_per_predictor_rank=5.0`. Then,
the nominal rank becomes 10, and the search evaluates only the eight compatible pairs
$(h,r_\pi)=(1,10),\ldots,(8,10)$. In the figure below, the full feasible domain is left neutral and
only those EPV pairs are colored by their actual mean response-standardized CV-MSE under the same
validation splits used in the exhaustive example.

![Pulp EPV search domain](assets/generated/pulp/epv_search_domain.svg)

There is no conditional predictor-rank selection in this case: $r_\pi$ is already fixed, and only
the component path remains to be selected.

### Other predictor-rank policies { #predictor-rank-policies }

A one-element explicit `predictor_rank_values` sequence also fixes one rank and therefore has no
conditional predictor-rank selection. A longer explicit sequence restricts the predictor-rank
domain and optimizes only over those supplied ranks, while `predictor_rank_values=None` uses the
full feasible integer domain described above. `n_components_values` restricts the component counts
and `max_predictor_rank` restricts the upper predictor-rank boundary.

## Adaptive search and scoring { #scoring-and-conditioned-path-selection }

The examples above use exhaustive coverage across the set of specified predictor ranks. When the predictor-rank domain is large,
`search_method="adaptive"` can reduce the number of ranks evaluated at each $h$. The admissible
domain and the conditional predictor-rank selection rule are unchanged; only candidate coverage
changes.

For each $h$, adaptive search begins with a sparse deterministic set of predictor ranks and refines
around the best evaluated score. If the lower boundary of the tolerance-qualified region lies
between evaluated ranks, that interval is refined as well. Small remaining intervals are evaluated
exhaustively. Ranks that are never evaluated have no associated CV score and make no contribution
to selection.

`cv_results_` records the candidates actually evaluated, and `search_is_exhaustive_` reports whether
those candidates happened to cover the complete admissible domain. `predictor_rank_profile(h)` and
its reference and retained selections likewise describe the evaluated ranks.

The figures use the default scorer, `"neg_response_standardized_mse"`; maximizing it is equivalent
to minimizing the response-standardized CV-MSE shown above. Other scikit-learn scorer names,
scorer callables, and `scoring=None` are also accepted. With another scorer, conditional
predictor-rank selection is performed on that configured-score scale.

## Reference details

The sections below collect search configuration, stored evidence, and refitting details. They use the
same domain and selection rules defined above.

### Pipelines and fold-local preprocessing { #pipelines-and-fold-local-preprocessing }

The search estimator may be a direct `PiPLSRegression` or a scikit-learn `Pipeline` whose final
step is `PiPLSRegression`. The complete estimator is cloned and fitted inside every training split,
so learned preprocessing must remain inside the estimator or pipeline.

For each candidate, `PiPLSSearchCV` replaces the terminal estimator's `n_components` and
`predictor_rank`. Other estimator settings, including scaling, response-subspace policy,
`svd_solver`, and `random_state`, are retained.

### Cross-validation protocols and metadata { #cross-validation-protocols-and-metadata }

`PiPLSSearchCV` accepts scikit-learn-compatible splitters and explicit split iterables, including
grouped, repeated, predefined, and temporal protocols. `groups` is an explicit `fit()` parameter
and participates in scikit-learn metadata routing when routing is enabled and requested.

Foldwise $R^2$ requires at least two validation observations per split.

### Split variation

`component_path_.cv_mse_std` is the population standard deviation of the split-specific
response-standardized MSE values. Maintained path and predictor-rank-profile figures show symmetric
$\pm 1$ SD bars. These bars describe split-to-split variation; they are not confidence intervals and
do not enter selection.

### Search-owned selection rules { #search-owned-selection-rules }

`PiPLSSearchCV.select()` returns one immutable stored component-path row without fitting or mutating
the search. `rule="minimum_cv_mse"` applies the component-path tolerance described above;
`rule="best_score"` instead selects the maximum configured-score row. A component count may also be
selected directly. In every case, the predictor rank is the rank already retained conditionally at
that $h$.

### Final-model refitting and provenance { #final-model-refitting }

`refit()` fits exactly one stored component-path row on the supplied full data. Exactly one of an
existing `selection`, a named `rule`, or `n_components` is required. The search clones the configured
estimator or pipeline, inserts the selected $(h,r_\pi)$ pair, fits the clone, and attaches the
immutable
selection as `model.selection_`.

The returned estimator owns the fitted model. The search object retains the candidate and selection
evidence and is not mutated by `refit()`.

### Computational consequences { #computational-consequences }

If $N_{\mathrm{split}}$ validation splits and $N_{\mathrm{pair}}$ candidate pairs are evaluated, the
candidate search performs

\begin{equation}
N_{\mathrm{candidate\ fit}}=N_{\mathrm{pair}}N_{\mathrm{split}}
\end{equation}

fold-local fits, in addition to feasibility probes. A final `refit()` adds one full-data fit.
Restricting the component or predictor-rank domain, using adaptive coverage, or changing the number
of validation splits changes the search evidence rather than merely its execution cost.

`n_jobs` controls parallel execution of candidate pairs. More workers can reduce elapsed time but
increase peak memory. See [Troubleshooting](troubleshooting.md#the-path-search-is-too-slow-or-uses-too-much-memory)
for practical controls.

### Result objects and scoring functions

`component_path_`, `select()`, and `predictor_rank_profile()` expose immutable records of the path,
retained selection, and conditional predictor-rank evidence.

::: pipls.component_path.PiPLSComponentPath
    options:
      show_signature: false
      members: false

::: pipls.component_path.PiPLSSelection
    options:
      show_signature: false
      members: false

::: pipls.component_path.PiPLSPredictorRankProfile
    options:
      show_signature: false
      members: false

::: pipls.component_path.PiPLSPredictorRankEvidence
    options:
      show_signature: false
      members: false

The positive loss and its scikit-learn-oriented negative scorer use the same fold-local
response-standardized MSE definition used in the exhaustive example.

::: pipls.metrics.response_standardized_mse
    options:
      members: false

::: pipls.metrics.neg_response_standardized_mse
    options:
      members: false
