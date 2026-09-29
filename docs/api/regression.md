# PiPLSRegression

`PiPLSRegression` fits exactly one fixed pair $(h,r_\pi)$: `n_components` is the number of
paired latent modes $h$, while `predictor_rank` is the retained predictor-subspace dimension
$r_\pi$. Both are required. The estimator performs no cross-validation or parameter selection; use
[`PiPLSSearchCV`](path.md) when either rank control is to be selected from data.

The default `response_subspace="cross_covariance"` is the construction used in the
[peer-reviewed companion publication](../citation.md#companion-paper). The alternative
`"least_squares"` policy is a least-squares/RRR-inspired software extension. Their mathematical
criteria are defined under [Response-subspace selection](../theory.md#response-subspace-selection).
`scale_x` and `scale_y`, when specified, override `scale` independently for predictors and
responses; centering remains part of every fit.

::: pipls.PiPLSRegression
    options:
      members:
        - fit
        - predict
        - transform
        - fit_transform
        - inverse_transform
        - score
        - get_feature_names_out
        - set_output

## PiPLSDecomposition

A fitted estimator exposes its Π-PLS-specific factorization as the immutable `decomposition_`
result. It contains the orthonormal predictor directions $\mathbf{P}$, dilations $D_k$, orthonormal
response directions $\mathbf{Q}$, and predictor-rank diagnostics. Its
`standardized_regression_map` property returns $\mathbf{P}\mathbf{D}\mathbf{Q}^{\mathsf T}$ in
centered/scaled coordinates. See [Canonical terminology](../theory.md#canonical-terminology) for
the distinction between these directions and reconstruction loadings.

::: pipls.decomposition.PiPLSDecomposition
    options:
      show_signature: false
      members:
        - standardized_regression_map

## PredictorRankSupportWarning

A direct fixed fit emits `PredictorRankSupportWarning` when $n/r_\pi<3$. The warning is diagnostic:
it does not alter the requested rank, while algebraically or numerically infeasible ranks remain
errors. Search-time predictor-rank support policies are defined under
[Predictor-rank policies](../path_selection.md#predictor-rank-policies).

::: pipls.PredictorRankSupportWarning
    options:
      members: false
