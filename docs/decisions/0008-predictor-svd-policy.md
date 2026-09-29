# Decision 0008: predictor SVD solver policy

## Status

Accepted and implemented. Decision 0155 preserves the predictor-only randomized-SVD boundary
and generalizes the exact response-side factorization contract across both implemented
response-subspace policies.

## Context

The predictor decomposition can dominate Pi-PLS cost when both the sample count and feature count
are large. Predictor-rank search policy and linear-algebra approximation must remain separate so a
user can independently request exhaustive versus adaptive rank search and exact versus approximate
predictor SVD.

## Decision

`PiPLSRegression` exposes:

```python
svd_solver="auto"       # "full", "randomized", or "auto"
random_state=0
```

Only the first SVD of the centered/scaled predictor matrix may be randomized. Response-side
orthogonal factorizations and the final SVD of `W` remain exact because they are normally much
smaller and define the response subspace and final Pi-PLS coupling. The cross-covariance policy
uses exact SVD of `Z.T @ Y`; the least-squares policy uses exact reduced QR of `Z` followed by
exact SVD of `Q_Z.T @ Y`.

The automatic solver uses randomized SVD exactly when all three conditions hold:

```python
min(n_samples, n_features) >= 500
n_samples * n_features >= 1_000_000
predictor_rank <= 0.2 * min(n_samples, n_features)
```

Otherwise it uses the full thin SVD. Explicit `"full"` and `"randomized"` choices override the
automatic rule.

`random_state` accepts an integer seed, a NumPy `RandomState`, or `None`, under the current
public validation contract. The default value `0` makes the default estimator reproducible.
Candidate models in path CV receive the same solver policy and cloned random-state parameter.

The resolved solver and numerical-rank diagnostics are canonical fields of `decomposition_`:
`predictor_svd_solver`, `x_rank`, `x_rank_is_exact`, and `rank_tolerance`. Under full SVD, `x_rank`
is the complete numerical rank. Under randomized truncated SVD, it is only the verified number of
retained singular values above the tolerance and `x_rank_is_exact` is false.

## Relationship to Decision 0155

Decision 0155 does not broaden `svd_solver`: only the predictor decomposition that constructs the
retained predictor basis may be randomized. The cross-covariance route therefore keeps exact SVD
of `Z.T @ Y`, while the least-squares route keeps exact reduced QR of `Z` and exact SVD of
`Q_Z.T @ Y`; the final SVD of `W` is exact under both policies. The durable contract is that
response-subspace selection and final latent coupling are exact, while randomized SVD remains a
predictor-side performance option only.

## Consequences

- `svd_solver="full"` remains the theory and regression-test reference path.
- `svd_solver="randomized"` is available for explicit performance control on smaller test
  problems and large production problems.
- `svd_solver="auto"` is conservative and does not randomize moderate matrices or ranks that are
  large relative to the smaller matrix dimension.
- Exact and approximate solver choices can produce small numerical differences even with identical
  predictor-rank search results.
- Randomized-SVD accuracy and reproducibility require dedicated tests and diagnostics.
