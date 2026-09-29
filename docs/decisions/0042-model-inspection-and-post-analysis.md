# Decision 0042: model inspection and post-analysis architecture

## Status

Accepted, with ordinary-PLS comparison ownership refined by Decision 0045 and rendering ownership
refined by Decision 0083. The
separation of selection diagnostics, fitted-model interpretation, prediction diagnostics,
immutable inspection results, optional plotting, explicit labels, physical axes, and canonical
artifacts remains in force. The corrective API, numbered-example, artifact, and boundary-test
migration required by Decision 0045 is complete.

## Context

Before this architecture was implemented, the real-data workflows answered a model-development
question through canonical Pi-PLS and ordinary PLS component-path CSV files and a CSV-derived
comparison PDF, then fitted one fixed Pi-PLS model chosen through a visible component-count
constant. The maintained numbered series later isolated comparison from Pi-PLS-only real-data
analysis. Those artifacts were selection diagnostics but did not yet provide supported analysis
of the fitted factorization, shared PLS-family latent structure, predictions, or residuals.

The companion Pi-PLS implementation contains a combined display of the predictor rotations $P$,
the dilation matrix $D$, the response rotations $Q$, standardized observed-versus-predicted values,
and residuals. It also distinguishes line displays for spectra from bar displays for a small set of
scalar variables. These ideas are useful, but the package needs clearer separation between
full-data interpretation and predictive validation, and it must not retain training data merely to
support plotting.

Established PLS analysis also includes scores, loadings, regression coefficients, biplots, and
observation diagnostics. The package should demonstrate a restrained subset of these tools where
they are scientifically readable, without turning plots into automatic variable-selection or
significance procedures.

## Decision

The repository distinguishes three analysis stages:

1. **Model-selection diagnostics.** Immutable component paths, conditional predictor-rank
   profiles, and caller-rendered figures describe the path over component counts and predictor
   ranks. `examples/_support/pls_component_path.py` remains an example-local ordinary-PLS comparison
   evaluator.
2. **Fixed-model interpretation.** A full-data fitted Pi-PLS model provides the method-specific
   factorization and the shared PLS-family score, loading, coefficient, biplot, and observation
   quantities. These plots describe the fitted model and are not validation results. Ordinary PLS
   remains a component-path comparator rather than a second post-analysis model.
3. **Prediction diagnostics.** Observed-versus-predicted and residual displays use predictions
   supplied explicitly by the caller. Every diagnostic records whether the values are fitted,
   fixed-parameter out-of-fold, selection-conditioned out-of-fold, or external-test predictions.

Reusable numerical analysis belongs in the public `pipls.inspection` submodule. It contains pure
NumPy computations and immutable result objects, with no pandas or Matplotlib dependency. Plotting
and report composition are caller-owned under Decision 0083; the runtime package exposes
no plotting submodule or public `plot_*` convenience functions.

Dataset-specific orchestration, explicit data reading, variable-name acquisition, fixed model
choices, out-of-fold prediction loops, table construction, artifact writing, and multipage-report
composition remain under `examples/`. Repository-maintenance and data-preparation code remains
under `scripts/` or the relevant dataset directory; it is not mixed with user analysis.

### Pi-PLS interpretation contract

For the centered and scaled regression map,

\begin{equation}
B_{\mathrm{cs}} = P D Q^\mathsf{T}.
\end{equation}

The primary Pi-PLS decomposition display contains:

- the predictor rotations $P$;
- the dilation values $D_k=D_{kk}$;
- the dilation-weighted response rotations $D_k q_{jk}$, represented by $QD$.

The display calls $P$ predictor rotations or predictor directions, not ordinary PLS loadings.
`PiPLSRegression.x_rotations_` equals $P$, whereas `x_loadings_` is a separately calculated score
reconstruction quantity.

Component signs may be canonicalized on copied arrays for stable display. For each component, the
largest-magnitude entry of the corresponding predictor rotation is made nonnegative, and the same
sign is applied to the response rotation. The transformation must preserve

\begin{equation}
P_{\mathrm{display}}D Q_{\mathrm{display}}^\mathsf{T}=PDQ^\mathsf{T}.
\end{equation}

The fitted estimator and its read-only `decomposition_` arrays must not be mutated.

Predictor rendering is selected explicitly by the example or caller:

- `"bar"` for a small set of scalar predictors;
- `"line"` for an ordered physical axis such as wavelength or wavenumber.

Categorical predictor and response displays require caller-supplied scientific labels. Real-data
examples read `X.csv` and `Y.csv` headers visibly when those headers contain the relevant names; a
programming user may instead supply labels from a schema or any other explicit metadata source.
Maintained renderers use side-by-side bars for categorical variables and overlaid lines for a
physical predictor axis. Numerical inspection does not read files, infer whether predictors are
spectra, generate substitute variable names, smooth or interpolate supplied curves, or reorder a
supplied physical axis.

### Prediction-diagnostic contract

Prediction diagnostics accept observed and predicted responses explicitly. The estimator does not
retain its training observations for plotting.

Residuals use

\begin{equation}
e_{ij}=y_{ij}-\hat y_{ij}.
\end{equation}

For a combined multivariate display, standardization is a display transformation based on the
supplied observed responses:

\begin{equation}
z_{ij}=\frac{y_{ij}-\bar y_j}{s_j},\qquad
\hat z_{ij}=\frac{\hat y_{ij}-\bar y_j}{s_j},\qquad
e^{(z)}_{ij}=z_{ij}-\hat z_{ij},
\end{equation}

where $s_j$ is the sample standard deviation calculated with `ddof=1`. Constant response columns
are rejected because their standardized diagnostics are undefined.

Prediction diagnostics also include the response-wise coefficient of determination

\begin{equation}
R_j^2 = 1 -
\frac{\sum_i (y_{ij}-\hat y_{ij})^2}
     {\sum_i (y_{ij}-\bar y_j)^2}.
\end{equation}

The provenance label governs its interpretation: fitted-value $R^2$ is descriptive training-fit
evidence, while OOF or external-test $R^2$ describes those supplied predictions. Prediction
displays may include standardized observed-versus-predicted values, standardized residuals,
response-wise standardized RMSE, response-wise $R^2$, and a caller-owned pooled standardized
residual histogram with a matched normal reference. Such a histogram is descriptive rather than a
formal normality test.

Real-data examples may calculate fixed-parameter out-of-fold predictions after the user has chosen
component counts from the same observations. Such results are labeled
`selection-conditioned OOF predictions`; they are descriptive diagnostics, not unbiased nested-CV
or external-test estimates.

The maintained complete real-data workflows use response-wise selection-conditioned OOF $R^2$ as
the preferred visible scalar prediction diagnostic while retaining standardized RMSE in the public
numerical result. Response-wise OOF $R^2$ is calculated from the row-ordered OOF predictions after
any repeated held-out predictions have been combined per observation; it is not an average of
fold-wise $R^2$ values. Maintained response-wise $R^2$ bar plots place the upper limit at exactly
1.0, never place the lower limit above 0.0, extend below negative values rather than clipping them,
and show the zero reference. Fitted-value and OOF $R^2$ remain explicitly distinguished by
prediction provenance.

### Shared PLS-family analysis contract

Decision 0045 supersedes the restriction to a concrete ordinary `PLSRegression` model. The shared
PLS-family surface uses compatible public fitted quantities from either `PLSRegression` or
`PiPLSRegression`:

- X scores;
- X loadings;
- Y loadings;
- regression coefficients;
- raw score-distance and X-reconstruction-residual diagnostics for suitable larger data.

Scores, loadings, and coefficients support explicit component or response subsets. Predictor
loadings and coefficients use the same explicit bar-versus-line rendering contract as Pi-PLS
predictor directions.

The raw score distance uses the fitted training-score center and covariance, with the
Moore--Penrose inverse for numerical rank deficiency. The X reconstruction residual uses the
public transform/inverse-transform round trip. The initial implementation draws no theoretical
limits and performs no automatic observation labeling.

Biplot scaling must be mathematically stated and must preserve the selected score-loading
reconstruction. Biplots are demonstrated only where the number of predictor arrows is readable;
they are not generated for the high-dimensional spectral datasets.

Observation diagnostics initially report raw descriptive quantities only. Theoretical probability
limits, confidence ellipses, VIP scores, automatic variable selection, uncertainty intervals,
permutation tests, and contribution plots require separate decisions and are not implied by this
architecture.

### Plotting and artifact contract

Plotting functions:

- accept computed results and explicit labels rather than retrieving hidden data;
- return the Matplotlib `Figure` and named axes;
- do not call `show()`;
- do not save files;
- do not mutate estimators or input arrays;
- validate dimensions, labels, selected components, and selected responses.

For real-data analyses, numerical CSV files are canonical and PDFs are views reconstructed from
those tables. Prediction tables retain original-unit and standardized observed values, predictions,
residuals, sample identity, response identity, and prediction provenance. The numbered reports
contain one selected Pi-PLS model, so no redundant model column is stored. Pi-PLS
response-direction tables retain $q_{jk}$, $D_k$, and $D_kq_{jk}$ separately rather than exporting
only the plotted product. Pickled estimators are not analysis artifacts.

## Implementation sequence

Implementation proceeds as a series of small patches:

1. pure Pi-PLS display-factor and prediction-diagnostic computations — complete;
2. Pi-PLS decomposition and prediction plots — complete;
3. shared score, loading, coefficient, biplot, and observation analysis under
   estimator-neutral shared names — complete;
4. example-local fixed-model OOF and post-analysis artifact helpers, integrated first with
   Pulp — complete;
5. Sugarcane spectral line analysis — complete;
6. Tobacco pagination and observation diagnostics — complete;
7. the Pulp biplot and a final cross-dataset analysis-surface review — complete.

The temporary standalone `09_model_inspection.py` demonstration was removed after the complete
Pulp, Sugarcane, and Tobacco workflows superseded it. The reusable package APIs remain documented
directly and exercised through focused tests and the complete real-data examples.

The package numerical core, estimator fitting, path-selection engine, and current component-path
helpers do not change as part of this decision.

## Consequences

- Component-path plots remain selection diagnostics rather than being repurposed as post-fit
  interpretation.
- Full-data decomposition, score, loading, and coefficient plots are described as fitted-model
  views and do not imply predictive validation.
- Prediction provenance is part of every prediction-diagnostic result and artifact; complete
  workflows keep selection-conditioned OOF diagnostics distinct from fitted-value diagnostics of
  the final model trained on all development observations.
- Pi-PLS-specific $P$, $D$, and $Q$ interpretation is supported without conflating rotations with
  ordinary PLS loadings.
- Low-dimensional scalar and high-dimensional spectral examples can share one numerical
  inspection surface while making their caller-owned rendering choice explicit.
- Optional rendering dependencies do not become core installation requirements.
- Advanced PLS diagnostics remain deferred until their definitions and interpretation boundaries
  are reviewed separately.
