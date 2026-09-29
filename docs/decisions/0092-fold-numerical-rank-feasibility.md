# Decision 0092: cap path selection by fold numerical rank

## Status

Accepted and implemented. Decision 0154 preserves fold numerical-rank preflight, removes the
$n/c$ term from the general search ceiling, and confines it to the explicit EPV policy.

## Context

`PiPLSSearchCV` previously bounded predictor rank by total-sample support, transformed feature count,
and centered training-fold dimensions. The fixed Pi-PLS core separately rejects a requested rank
above the numerical rank of its preprocessed predictor matrix. A rank-deficient training fold could
therefore make one otherwise admissible candidate abort the complete path search, even when lower
ranks were valid.

Rank deficiency is an expected numerical condition. The search meta-estimator should construct only
candidates that every materialized training fold can fit, including when a supported pipeline
changes the predictor representation inside each fold.

## Decision

Determine the path ceiling before candidate evaluation from the minimum verified predictor rank
across the materialized training folds.

1. For each training fold, fit any pipeline preprocessing only on that fold and obtain the
   predictors supplied to the terminal `PiPLSRegression`.
2. Apply the terminal estimator's predictor centering/scaling and run the same private predictor-SVD
   decomposition used by fixed fitting at the fold's centered algebraic rank ceiling. Preserve the
   configured predictor-SVD policy and random-state parameter; response-side Π-PLS work is not part
   of feasibility preflight.
3. Fixed fitting still reports rank infeasibility through the private core error carrying the
   requested rank, verified rank, exactness flag, and tolerance. Search preflight reads the same
   predictor decomposition evidence directly. The public fixed-estimator failure remains a
   `ValueError` with the existing numerical-rank message.
4. Let $r_{\mathrm{num,min}}$ be the minimum verified rank across folds. The hard search ceiling is

   \begin{equation}
   r_{\pi,\mathrm{hard}}=\min\left[p_{\min},n_{\mathrm{train,min}}-1,
   r_{\mathrm{num,min}}\right].
   \end{equation}

   An integer `max_predictor_rank` is an additional user upper bound. The explicit EPV policy first
   computes its full-sample nominal $\min[p,\lceil n/c\rceil]$ rank and then clips it by this hard
   ceiling and any user maximum.
5. Validate explicit component and predictor-rank sequences against this resolved ceiling before
   scoring any candidate. Do not add failed or nonfinite candidate rows to `cv_results_`.
6. If any fold has no positive verified predictor rank, fail clearly and transactionally.
7. Predictor-rank preflight must not advance shared NumPy random state. For `random_state=None`, use
   a local copy of the global state across the fold probes; explicit `RandomState` parameters remain
   isolated by the existing fold-local estimator cloning. Later candidate fits therefore retain
   their established random-state semantics. Unrelated preprocessing warnings remain visible.

For explicit randomized SVD, the verified rank remains a retained-rank lower bound rather than a
claim of exact full numerical rank. The minimum of those verified bounds is therefore the
conservative path ceiling.

## Consequences

Default and explicit path searches now treat rank-deficient predictors as an ordinary bounded
search case. `max_predictor_rank_` records the effective hard/user ceiling, and
`n_components_values="all"` resolves only component counts supported by the active predictor-rank
policy.

The preflight adds one predictor decomposition per training fold and one fold-local preprocessing
fit for pipelines. It does not construct a response subspace or complete a dummy Π-PLS fit.
Candidate evaluation, OOF generation, and refit behavior are otherwise unchanged.
