# Decision 0148: predictor-rank tolerance selection

## Status

Accepted and implemented. The constructor tolerance controls, immutable optimized-rank evidence,
conditioned-path selection rules, and adaptive tolerance-boundary refinement are current. Decision
0154 owns the explicit EPV policy and full-domain automatic-search contract.

## Context

`PiPLSSearchCV` makes two model-complexity choices at different levels. For each evaluated component
count $h$, it first chooses a retained predictor rank $r_{\pi}$. It then builds `component_path_`
from those conditional choices and selects a component count from that path.

The second choice already has public relative and absolute CV-MSE tolerances through Decision 0146.
Before this decision, the first choice maximized `mean_test_score` and used private `rtol=1e-12`
and `atol=1e-15` values only to identify numerical ties. Those private constants remain appropriate for
stable equality handling, ranking, and adaptive-search refinement, but they do not provide a
user-controlled tradeoff between predictor-subspace dimension and observed validation performance.

The two choices therefore need separate tolerance controls. Predictor-rank tolerances belong to
search construction because they determine every row of `component_path_`. Component-count
tolerances remain post-fit arguments to `select()` and `refit()` because they act on the completed
path.

## Decision

### Use hierarchical parsimony

The selection procedure is explicitly hierarchical:

1. evaluate predictor-rank candidates independently for every component count $h$;
2. select one tolerance-qualified retained rank $r_{\pi}$ at each $h$;
3. construct `component_path_` from those conditionally retained candidates;
4. select `n_components` from that conditioned path by `best_score`, `minimum_cv_mse`, or manual
   component-count lookup.

Predictor-rank tolerance does not select a component count, and component-count tolerance does not
revisit predictor ranks.

### Define predictor-rank tolerances in configured-score units

For fixed $h$, let $S_{hr}$ be the mean configured cross-validation score for evaluated predictor
rank $r$, and let

\begin{equation}
S_{h,\max}
=
\max_r S_{hr}.
\end{equation}

For resolved nonnegative predictor-rank tolerances $\delta_{\mathrm{rel},r}$ and
$\delta_{\mathrm{abs},r}$, rank $r$ qualifies only when both

\begin{equation}
S_{hr}
\geq
S_{h,\max}
-
\delta_{\mathrm{rel},r}
\left|S_{h,\max}\right|
\end{equation}

and

\begin{equation}
S_{hr}
\geq
S_{h,\max}
-
\delta_{\mathrm{abs},r}.
\end{equation}

Equivalently, the effective configured-score threshold is

\begin{equation}
T_{S,h}
=
\max\left[
S_{h,\max}
-
\delta_{\mathrm{rel},r}
\left|S_{h,\max}\right|,
S_{h,\max}
-
\delta_{\mathrm{abs},r}
\right].
\end{equation}

The smallest evaluated predictor rank satisfying $S_{hr}\geq T_{S,h}$ is retained. The rule applies
to positive, negative, or zero reference scores and preserves the configured scorer orientation that
larger values are better.

With the default scorer, let $M_{hr}=-S_{hr}$ and $M_{h,\min}=\min_r M_{hr}$. The same conditions are

\begin{equation}
M_{hr}
\leq
\left(1+\delta_{\mathrm{rel},r}\right)M_{h,\min}
\end{equation}

and

\begin{equation}
M_{hr}
\leq
M_{h,\min}
+
\delta_{\mathrm{abs},r}.
\end{equation}

Thus the default-scorer interpretation matches component-count tolerance selection, while custom
scorers retain their own units and meaning.

### Add separate constructor parameters

`PiPLSSearchCV` gains keyword-only constructor parameters:

```python
PiPLSSearchCV(
    predictor_rank_relative_tolerance=None,
    predictor_rank_absolute_tolerance=np.inf,
)
```

`predictor_rank_relative_tolerance=None` resolves to
`sqrt(np.finfo(np.float64).eps)`. A supplied value must be a finite nonnegative real number.
`predictor_rank_absolute_tolerance` must be a nonnegative real number or positive infinity; positive
infinity disables the absolute cap. Booleans, NaN, negative values, and negative infinity are
invalid.

The arguments are not added to `PiPLSRegression`, `select()`, or `refit()`. They determine the fitted
search evidence and therefore participate in scikit-learn cloning, `get_params()`, `set_params()`,
repr, and pickle behavior.

Nondefault predictor-rank tolerances are rejected when `predictor_rank_values="epv"` or a
one-element fixed rank sequence makes predictor-rank optimization inapplicable. Defaults remain
accepted for those policies and produce no tolerance provenance. `predictor_rank_values=None` and
multi-rank explicit sequences use the optimized policy and apply the tolerances.

### Refine adaptive coverage at the parsimony boundary

Private numerical tie handling remains separate from public parsimony tolerance. The existing
private tie constants continue to govern:

- `rank_test_score` equality;
- identification of the exact configured-score reference optimum;
- deterministic low-rank ordering among exact numerical ties;
- adaptive-search refinement around the exact evaluated optimum.

For `search_method="adaptive"`, first complete exact-reference refinement under Decision 0007.
Then let $r_b$ be the smallest evaluated rank satisfying both public tolerance caps. If the
immediately lower evaluated rank $r_a$ fails the tolerance and unevaluated admissible ranks remain in
$(r_a,r_b)$, refine that interval by deterministic midpoint bisection toward the tolerance crossing.
Use the same private five-rank exhaustive-switch threshold as exact-reference refinement, then retain
the smallest evaluated qualifying rank. If new evaluations change the exact evaluated reference
rank, complete exact-reference refinement around the new reference and resolve the tolerance bracket
again.

Adaptive candidate coverage may therefore depend on predictor-rank tolerance.
`search_method="exhaustive"` evaluates every admissible rank and returns the smallest admissible
qualifying rank directly. Candidate-level `cv_results_`, split scores, mean scores, CV-MSE summaries,
timing results, `rank_test_score`, and `search_is_exhaustive_` remain descriptions of the candidates
actually evaluated and are not rewritten by tolerance selection.

### Make predictor-rank provenance public and immutable

Add public `PiPLSPredictorRankEvidence` in `pipls.component_path`. It is not added to the narrow
package-level `pipls` namespace. The immutable record contains:

```python
reference_predictor_rank
reference_mean_test_score
reference_cv_mse_mean
reference_cv_mse_std
relative_tolerance
absolute_tolerance
```

and derives `score_threshold`. The selected candidate values remain on the containing
`PiPLSSelection`, path row, or profile and are not duplicated in the evidence record.

Expose the record through:

```python
selection.predictor_rank_evidence
profile.predictor_rank_evidence
path.predictor_rank_evidence
```

For an optimized `PiPLSComponentPath`, `predictor_rank_evidence` is an immutable sequence aligned
one-for-one with path rows. For fixed and EPV policies it is `None`. Every optimized path row,
including a row used as the component-count `reference_minimum`, retains its own predictor-rank
evidence.

`PiPLSPredictorRankProfile` additionally exposes:

```python
profile.reference_selection
profile.selection
```

`reference_selection` is the exact configured-score optimum under the private numerical tie rule
and contains no nested predictor-rank evidence. `selection` is the smallest evaluated rank satisfying
the public tolerances and carries `predictor_rank_evidence`.

All new records and fields follow the existing direct-construction, defensive-copy, read-only,
validation, equality, and pickle-revalidation contracts.

### Define named component-count selection on the conditioned path

After predictor-rank tolerance selection, `component_path_` is the sole path used by component-count
rules:

- `best_score` returns the maximum configured-score row on the conditioned component path, with
  deterministic smaller-component and smaller-rank ordering for exact numerical ties;
- `minimum_cv_mse` applies Decision 0146 to the conditioned component path;
- manual `n_components` lookup returns the conditioned row stored at that count.

A global candidate in `cv_results_` can have a better score than the retained rank at its component
count when a nonzero predictor-rank tolerance favors a smaller rank. Such an unretained candidate is
not eligible for a named component-count rule. This change preserves the meaning of hierarchical
selection and must be explicit in user documentation.

The two relative tolerances are independent. With the default scorer, applying 10% predictor-rank
tolerance and then 10% component-count tolerance does not define one global 10% bound. In the
limiting case, the selected candidate can have CV-MSE up to

\begin{equation}
1.1^2 M_{\min}=1.21M_{\min}
\end{equation}

relative to the global candidate minimum. This is the expected consequence of two sequential
parsimony decisions.

## Consequences

Users can request a smaller retained predictor subspace under an explicit bounded validation-score
allowance, independently of the later component-count choice. The path remains deterministic and
fully inspectable; adaptive coverage may add tolerance-boundary evaluations, while custom scorers
keep their native orientation and units. The additional provenance makes both stages of parsimony
reconstructable from public immutable results.
