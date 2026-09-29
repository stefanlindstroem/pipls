# PiPLSSearchCV

Use `PiPLSSearchCV` when either of the two Π-PLS rank controls is to be selected from data. The
search evaluates
admissible `(n_components, predictor_rank)` pairs by cross-validation; `n_components` is the paired
latent-mode count $h$, while `predictor_rank` is the retained predictor-subspace dimension
$r_\pi$. Use [`PiPLSRegression`](regression.md) when both controls are already fixed.

`fit()` evaluates and stores the search evidence; it does not produce a prediction model. After
fitting, `select()` retrieves one immutable stored row, `oof_report()` evaluates one existing
selection on the stored validation splits, and `refit()` fits one selected $(h,r_\pi)$ pair on the
supplied full data. The search object does not delegate prediction methods or retain the model or OOF report
returned by these operations.

Candidate-estimator settings such as scaling, response-subspace policy, solver choice, and random
state belong to the supplied `PiPLSRegression` estimator or terminal pipeline step. Search replaces
only `n_components` and `predictor_rank` on cloned candidates. With `estimator=None`, ordinary
`PiPLSRegression` defaults are used.

Exact search-domain, predictor-rank-policy, scoring, validation, selection, provenance, and
computational contracts are defined in [Path and selection](../path_selection.md). Exact stored-split
reuse and OOF reporting contracts are defined in [OOF diagnostics](../oof_diagnostics.md).
Worked model-selection procedures belong to the [tutorials](../tutorials/synthetic.md) and
[examples](../examples.md).

::: pipls.PiPLSSearchCV
    options:
      members:
        - fit
        - select
        - refit
        - oof_report
        - predictor_rank_profile
