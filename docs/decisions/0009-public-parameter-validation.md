# Decision 0009: public parameter validation and statistical-support warning

Status: accepted and refined by Decisions 0039, 0153, and 0154. Fixed-estimator validation now
covers only explicit ranks and warns at $n/r_\pi<3$. Decision 0154 moves the low-$c$ search
warning to the explicit EPV policy and removes the `max_predictor_rank="rule"` sentinel; that
runtime migration is implemented.

## Context

`PiPLSRegression` exposes several integer or integer-like controls whose invalid values can
otherwise fail inside NumPy, joblib, or scikit-learn with inconsistent messages. The parameter $c$, exposed as `samples_per_predictor_rank`, controls the explicit
EPV heuristic. Decision 0154 specifies that this policy uses the total number of observations supplied
to `fit()`.

## Decision

Constructor parameters are validated at the start of `fit()` before array copying, cross-validation
materialization, or numerical decomposition.

- `n_components` and an integer `predictor_rank` must be positive Python or NumPy integers.
  Booleans and integral-valued floats are rejected.
- Integer `cv` values must be at least 2. NumPy integers are normalized to Python integers.
  Cross-validation splitter objects and explicit split iterables remain supported.
- `n_jobs` must be `None` or a nonzero Python or NumPy integer. Negative joblib values remain
  valid.
- `random_state` accepts an integer in the unsigned 32-bit seed interval $[0, 2^{32}-1]$,
  a NumPy `RandomState`, or `None`, under the current public API. The integer default `0` is
  reproducible; `None` follows NumPy's global random state.
- `scale` and `copy` accept only Python or NumPy booleans. `scale_x` and `scale_y` accept Python or
  NumPy booleans or `None`, where `None` inherits the value of `scale`.
- Invalid scalar `scoring`, `svd_solver`, and rank-mode values fail with package-level `ValueError`
  messages rather than incidental errors from dependencies.

When `samples_per_predictor_rank < 5` is used with `predictor_rank_values="epv"`, one public
`PredictorRankSupportWarning` is emitted per top-level search fit. The warning states that the
resulting EPV-fixed rank may not have sufficient statistical support to be trusted without external
validation. Outside EPV, `samples_per_predictor_rank` does not define candidate ranks; a nondefault
value is rejected so it cannot be mistaken for an automatic-search bound.

The rank-bound calculation saturates safely at the algebraic limit for extremely small positive
$c$, avoiding floating-point overflow.

## Consequences

- Invalid exposed controls fail early and consistently.
- NumPy scalar integers and booleans receive deliberate, tested treatment.
- Low-support rank rules remain permitted but cannot be used silently.
- Internal CV candidate fits do not repeat the warning because they use explicit integer ranks.
- API-level tests cover invalid, boundary, NumPy-scalar, and degenerate values exhaustively.
