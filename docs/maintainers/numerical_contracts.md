# Numerical contracts

## Linear algebra and rank

- Use thin SVDs and avoid forming a predictor covariance matrix when `p >> n`.
- Use `numpy.linalg.eigh` only for explicitly symmetrized matrices.
- Do not form explicit inverses. Use solves, least squares, SVDs, or documented pseudoinverses.
- The fixed core classifies predictor singular values with
  `tau_X = max(n, p) * eps64 * s_1`; requested predictor rank above the verified numerical rank
  raises `ValueError`.
- Repeated or nearly repeated singular values identify subspaces, not stable signed basis columns.
  Compare projectors, principal angles, singular values, regression maps, or predictions.
- Invalid dimensions are errors and are never silently clamped.

## Private response-subspace factorization

The fixed numerical core now supports two exact response-subspace constructions under Decision
0155. The public estimator does not expose the selector until Step 3; its current call path therefore
continues to use the default cross-covariance construction.

For `response_subspace="cross_covariance"`, the core forms `Z.T @ Y` and obtains the response basis
from its exact thin NumPy SVD. This is the peer-reviewed companion-publication construction and the
pre-Decision-0155 numerical compatibility reference.

For `response_subspace="least_squares"`, the core obtains an orthonormal basis of `col(Z)` by exact
reduced NumPy QR, projects the responses as `Q_Z.T @ Y`, and obtains the response basis from an exact
thin NumPy SVD of that projected response matrix. The implementation does not form
`Z.T @ Z`, does not invert or pseudoinvert a Gram matrix, and introduces no independent numerical-rank
tolerance. This least-squares/RRR-inspired construction is a software extension and is not part of
the peer-reviewed companion publication.

`svd_solver` governs only the predictor decomposition used to construct $\mathbf{\Pi}$. Selecting
`svd_solver="randomized"` may therefore randomize the predictor basis, but it does not replace QR or
either response-side SVD with randomized routines. The retained predictor-rank verification remains
the single rank-feasibility contract for `Z`; the response-subspace helpers do not add a second rank
threshold.

## Estimator preprocessing and finite behavior

Every fixed fit learns predictor and response means from its own training observations. Both blocks
are always centered. `scale` supplies the backward-compatible scaling default for both blocks;
non-`None` `scale_x` and `scale_y` values override predictor and response scaling independently.
Enabled scaling uses safe sample standard deviations with `ddof=1`; disabled scaling stores a unit
scale vector for that block. Search candidates learn these quantities independently inside each
training fold, and `refit()` learns them again from the supplied full training data.

Public fits are transactional. A failed fit removes old and partial fitted state. A failed search
refit leaves the fitted search unchanged because refitting operates on a fresh clone.

Finite float64 inputs must either produce finite public fitted/output arrays or raise a clear
exception. Range-safe fallbacks are allowed only where ordinary finite calculations overflow or a
nonconstant column underflows to zero scale. Public fixed fitting rejects magnitudes that would
make the core predictor-response product unrepresentable.

`copy=False` may reuse independent writable arrays, but read-only inputs and overlapping predictor
and response storage must be copied when needed for correctness.

## Search feasibility and candidate evaluation

The search materializes one split set and verifies predictor feasibility in every transformed
training fold. With:

- `p_min`: minimum transformed feature count;
- `n_train_min`: minimum materialized training-fold size;
- `r_num_min`: minimum verified fold numerical rank;

the hard predictor-rank ceiling is

```text
min(p_min, n_train_min - 1, r_num_min)
```

An explicit positive integer `max_predictor_rank` adds a user restriction by taking the minimum with
that hard ceiling. `max_predictor_rank=None` adds no statistical or heuristic cap.

With `predictor_rank_values=None`, the ordinary rank domain contains every integer from one through
that effective ceiling, subject to `n_components <= predictor_rank`. The constructor default
`search_method="exhaustive"` evaluates every admissible pair. `search_method="adaptive"` uses the
same domain but may evaluate only a deterministic subset; `search_is_exhaustive_` reports achieved
coverage. Private numerical tie behavior uses dedicated numerical tie tolerances and deterministic
smaller-rank ordering.

The EPV policy is separate from hard feasibility. With total sample count `n` and positive finite
`c = samples_per_predictor_rank`, its nominal fixed rank is

```text
min(p, ceil(n / c))
```

and the effective EPV rank is the minimum of that nominal value and the effective hard/user
ceiling. The ceiling operation and use of the full supplied `n` are normative.
`samples_per_predictor_rank=10.0` is the EPV default; values below 5 remain legal and emit
`PredictorRankSupportWarning`. A nondefault `samples_per_predictor_rank` is invalid outside
`predictor_rank_values="epv"`. EPV and one-rank explicit sequences perform no predictor-rank
optimization and carry no predictor-rank tolerance evidence.

Candidate pairs satisfy `1 <= n_components <= min(n_targets, predictor_rank)` and are fixed-model
clones. Fold-local preprocessing, scoring, and warning suppression must not leak validation data.
The component-count domain is resolved after the active predictor-rank policy so
`n_components_values="all"` cannot request a component count above a fixed or EPV rank.

Decision 0148 preserves private score comparison for exact-reference identification,
`rank_test_score`, and adaptive refinement. Its accepted final retained-rank rule uses separate
public tolerances. For fixed component count $h$ and exact reference score $S_{h,\max}$, evaluated
rank $r$ qualifies when both

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

The smallest evaluated qualifying rank is retained. Under exhaustive coverage the reference is over
the complete declared admissible rank domain, so this is also the smallest admissible qualifying
rank. Under adaptive coverage, first refine around the exact evaluated score optimum. If the smallest
evaluated qualifying rank and its immediately lower failing evaluated neighbor bracket unevaluated
admissible ranks, refine that tolerance boundary by deterministic midpoint bisection as a second
adaptive phase. Both phases use one shared private exhaustive-switch threshold of five admissible
ranks. If tolerance-boundary evaluation changes the exact evaluated reference rank, complete
exact-reference refinement around the new reference before resolving the tolerance boundary again.
Adaptive candidate coverage may
therefore depend on the public predictor-rank tolerances.

## Response-standardized MSE

For each response, residuals are divided by the sample standard deviation learned from the
candidate's training responses. The scalar loss is the uniform mean over observations and response
columns. Constant response columns use the package's safe training-scale contract.

Path CV-MSE statistics give every materialized validation split equal weight, including repeated CV
and unequal validation-set lengths:

```text
cv_mse_mean = np.mean(split_cv_mse)
cv_mse_std  = np.std(split_cv_mse, ddof=0)
```

A one-split path therefore has `cv_mse_std == 0`. `cv_mse_std` is descriptive population SD across
splits; no standard error is derived or exposed.

## Selection tolerances

Predictor-rank and component-count tolerances are separate stages. Predictor-rank tolerances are
configured on `PiPLSSearchCV` and act in configured-score units within each component count.
Component-count tolerances are supplied to `select()` or rule-based `refit()` and act on the
resulting conditioned path in CV-MSE units. `refit(selection=...)` rejects nondefault tolerances
because the immutable selection has already resolved that decision.

Let `M_min` be the exact minimum stored path mean. For a resolved nonnegative relative tolerance
`delta_rel` and nonnegative absolute tolerance `delta_abs`:

```text
T_rel = (1 + delta_rel) * M_min
T_abs = M_min + delta_abs
T     = min(T_rel, T_abs)
```

The selected row is the smallest stored component count with `cv_mse_mean <= T`. Both caps must
hold, exact boundary equality is eligible, and a zero minimum keeps the relative cap at zero.

`relative_tolerance=None` resolves to `sqrt(np.finfo(np.float64).eps)`. Relative tolerance must be
finite and nonnegative. Absolute tolerance may also be positive infinity, which disables that cap;
NaN, negative values, and negative infinity are invalid. Tolerances apply only to
`rule="minimum_cv_mse"`.

The component-count selection retains an unruled exact-minimum path row, resolved tolerances, and
a derived effective threshold. Under Decision 0148, that reference row retains its own independent
predictor-rank evidence. Direct component lookup and `best_score` have no component-count tolerance
provenance but still carry predictor-rank evidence for optimized policies. All provenance must remain
exactly reproducible for the shared OOF-report and full-data-refit compatibility validation.

## OOF reporting

`oof_report()` reuses defensive copies of every split materialized by the fitted search. Each
selected fixed pair is refitted on every training fold. Multiple validation predictions for one
observation are averaged. Counts record how many predictions contributed.

Coverage representation is exact:

- positive count: complete finite prediction row;
- zero count: complete NaN prediction row.

Pooled OOF R2 is computed only over covered observations and is `None` when fewer than two covered
rows make it undefined. The report does not rescore candidates or fit a full-data model.

Foldwise $R^2$ scoring is rejected when any validation split contains fewer than two observations.
This is a protocol-neutral scorer-domain check; it does not identify or attach provenance for the
validation protocol that produced the singleton split.

## Immutable public results

Array-valued public result records defensively copy arrays, make them read-only, and restore that
storage contract during pickle reconstruction. Positive counts/ranks, admissible component/rank
pairs, finite scores, nonnegative MSE/SD, aligned shapes, and OOF coverage are producer-owned
invariants enforced by the estimator, search, or inspection computation rather than duplicated by
arbitrary result-dataclass construction.

Inspection helpers must return finite derived arrays or raise `ValueError`. Display-factor sign
changes must preserve `P D Q.T`; balanced biplot scaling must preserve the selected `T P.T`
reconstruction; prediction residuals use `observed - predicted` and response standardization uses
observed-response sample scales with `ddof=1`.

## Numerical review

Every numerical change must state its comparison tolerance and add at least one boundary-case test.
When behavior should not change, compare before/after arrays or scalar results directly rather than
relying only on the test suite.
