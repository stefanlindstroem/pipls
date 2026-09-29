# Model inspection

`pipls.inspection` provides pure numerical inspection helpers. Results are immutable defensive
copies; the module does not render figures or write files. `pipls_display_factors()` is specific to
Π-PLS, while latent structure, biplot coordinates, observation diagnostics, and prediction
diagnostics use PLS-family quantities and can also accept compatible fitted PLS estimators.
Finite inputs either produce finite float64 inspection quantities or raise `ValueError` when a
requested derived quantity is not representable.

The [Pulp tutorial](tutorials/pulp.md#interpret-representative-fitted-model-plots) shows maintained
plots built from these results. Rendering remains caller-owned.

## Π-PLS factors

A fitted `PiPLSRegression` represents its centered and scaled regression map as

\begin{equation}
\mathbf{B}_{\mathrm{cs}}=\mathbf{P}\mathbf{D}\mathbf{Q}^{\mathsf T}.
\end{equation}

`pipls_display_factors()` returns copied factors with deterministic display signs. Applying the same
sign to paired columns of $\mathbf{P}$ and $\mathbf{Q}$ leaves the regression map unchanged. The
default convention makes the first largest-magnitude predictor entry nonnegative; a selected
response can instead anchor the orientation. These sign choices affect display only and do not
change predictions or imply a positive or negative physical effect.

### Predictor directions { #predictor-directions }

Columns of `predictor_directions` are orthonormal predictor directions $\mathbf{P}$ in the regression
factorization. They are not X loadings, which instead describe score reconstruction and are returned
by `latent_structure()`.

### Dilation { #dilation }

Each nonnegative value $D_k=D_{kk}$ scales paired latent mode $k$ and should be interpreted together
with the matching columns of $\mathbf{P}$ and $\mathbf{Q}$.

### Response directions { #response-directions }

Columns of `response_directions` are orthonormal response directions $\mathbf{Q}$ before dilation. Use explicit
response labels when comparing them.

### Weighted response directions { #weighted-response-directions }

`weighted_response_directions` combines response-side orientation and mode strength; column $k$ is
$D_kQ_{:k}$.

Theory: [Diagonal latent coupling](theory.md#diagonal-latent-coupling).

::: pipls.inspection.PiPLSDisplayFactors
    options:
      show_signature: false
      members:
        - n_components

::: pipls.inspection.pipls_display_factors
    options:
      members: false

## Latent structure

`latent_structure()` copies fitted X scores, X loadings, Y loadings, and original-unit coefficients
without recomputing, rescaling, or sign-adjusting them. They describe the supplied fitted model and
are not validation results.

### Scores { #scores }

Columns of `x_scores` locate observations in fitted latent coordinates. Proximity is descriptive;
it does not establish groups, confidence regions, or outliers.

### X loadings { #x-loadings }

Columns of `x_loadings` describe predictor participation in score reconstruction. They are not
regression coefficients or Π-PLS predictor directions, and component signs may reverse without
changing the model.

### Y loadings { #y-loadings }

Columns of `y_loadings` describe responses in the fitted latent representation. With response
scaling, they refer to the standardized fitted representation rather than original-unit responses.

### Regression coefficients { #regression-coefficients }

Rows of `coefficients` map predictors to responses in original units. Raw magnitudes are not directly
comparable across variables with different units, and correlated predictors may share coefficient
weight.

::: pipls.inspection.LatentStructure
    options:
      show_signature: false
      members:
        - n_components

::: pipls.inspection.latent_structure
    options:
      members: false

## Biplot coordinates { #score-loading-biplot }

`biplot_coordinates()` balances two score and X-loading columns while preserving their rank-two
reconstruction. For component $k$,

\begin{equation}
a_k=\sqrt{\frac{\lVert p_k\rVert_2}{\lVert t_k\rVert_2}},\qquad
\widetilde t_k=a_kt_k,\qquad
\widetilde p_k=\frac{p_k}{a_k}.
\end{equation}

Plot `sample_coordinates` for observations and vectors from the origin to `predictor_coordinates`
for predictors. Similar arrow directions indicate similar loading patterns in the displayed plane;
they are not causal effects or automatic importance measures.

::: pipls.inspection.BiplotCoordinates
    options:
      show_signature: false
      members: false

::: pipls.inspection.biplot_coordinates
    options:
      members: false

## Prediction diagnostics

`prediction_diagnostics()` takes observed and predicted responses explicitly and requires a
provenance label. Standardization uses centers and sample standard deviations calculated from the
observed responses and does not alter the original-unit predictions. Selection-conditioned OOF
predictions remain descriptive post-selection diagnostics rather than an independent estimate of
future performance.

### Observed versus predicted { #observed-versus-predicted }

Use `observed_standardized` and `predicted_standardized` with an identity reference, while keeping
`prediction_kind` visible so fitted, OOF, and external-test predictions are not conflated.

### Residuals versus predicted { #residuals-versus-predicted }

Use `predicted_standardized` against `residual_standardized` with a zero-residual reference.
Patterns can reveal systematic structure but do not supply formal uncertainty tests.

### Standardized RMSE { #standardized-rmse }

`standardized_rmse` compares response-wise prediction error with the observed spread of each
response. It is not generally equal to the mean fold-local standardized loss used during path
selection.

### Response-wise coefficient of determination { #response-r2 }

For response $j$,

\begin{equation}
R_j^2=1-\frac{\sum_i (y_{ij}-\hat y_{ij})^2}{\sum_i (y_{ij}-\bar y_j)^2}.
\end{equation}

Interpretation follows `prediction_kind`: fitted values describe training fit, while OOF or
external-test values describe the supplied predictions. $R_j^2$ alone is not evidence of unbiased
post-selection performance.

::: pipls.inspection.PredictionKind
    options:
      members: false

::: pipls.inspection.PredictionDiagnostics
    options:
      show_signature: false
      members: false

::: pipls.inspection.prediction_diagnostics
    options:
      members: false

## Observation diagnostics { #observation-diagnostics }

`observation_diagnostics()` returns squared score distance from the fitted training-score center and
squared X-reconstruction residual from the model's public transform/inverse-transform round trip.
Large values indicate observations that are distant in fitted score space, poorly reconstructed in
predictor space, or both. The package supplies no theoretical limits, automatic outlier labels, or
contribution diagnostics.

::: pipls.inspection.ObservationDiagnostics
    options:
      show_signature: false
      members: false

::: pipls.inspection.observation_diagnostics
    options:
      members: false

## Interpretation and rendering boundary

Fitted factors, scores, loadings, coefficients, biplots, and observation diagnostics describe one
fitted model. Prediction diagnostics describe the prediction source stated by their provenance.
None supplies uncertainty intervals, causal interpretation, automatic variable selection, or an
unbiased post-selection performance estimate.

The caller owns chart composition, scientific coordinates, labels, legends, layout, saving, and
closing. Predictor and response names may come from ordinary Python sequences or data-frame column
labels. For spectral plots, retain the physical coordinate in its existing order; Π-PLS does not
smooth, interpolate, or reorder it.
