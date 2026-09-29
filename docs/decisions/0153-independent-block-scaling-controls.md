# Decision 0153: independent predictor and response scaling controls

## Status

Accepted and implemented. This decision refines Decisions 0002, 0009, and 0025 without changing the
fold-local preprocessing boundary.

## Context

`PiPLSRegression` historically exposed one boolean `scale` parameter. It controlled both predictor
and response standardization after centering. That matches the common ordinary-PLS interface, but it
couples two statistically distinct choices.

A supported scikit-learn pipeline may learn predictor preprocessing before the terminal
`PiPLSRegression`. In particular, future block-aware predictor scaling must be fitted inside each
training fold. Applying Pi-PLS's ordinary columnwise predictor standardization after such a transform
can replace the relative scales learned by the upstream transformer. Setting `scale=False` avoids
that problem but also disables response standardization, even when response scaling should remain
part of the Pi-PLS model.

The downstream Pi-PLS implementation already stores and applies `x_scale_` and `y_scale_`
independently. The coupling therefore exists only in the public policy that decides whether those
vectors are learned.

## Decision

- Keep `scale: bool = True` as a backward-compatible default policy for both predictor and response
  scaling.
- Add `scale_x: bool | None = None` and `scale_y: bool | None = None` to `PiPLSRegression`.
- Resolve effective predictor scaling as `scale` when `scale_x is None`, otherwise use `scale_x`.
  Resolve response scaling analogously from `scale_y`.
- Continue centering both predictors and responses for every fit, regardless of scaling policy.
- When scaling is enabled for a block, learn safe sample-standard-deviation scales with the existing
  `ddof=1` convention. When disabled, store a unit scale vector for that block.
- Preserve the existing coefficient, intercept, transform, inverse-transform, and prediction
  algebra, which already maps predictor and response scales independently.
- Validate `scale_x` and `scale_y` as Python or NumPy booleans or `None`. Keep `scale` itself boolean
  so existing invalid-value behavior remains stable.
- Preserve fold-local ownership. A predictor transformer in a supported pipeline is cloned and fitted
  inside each training fold; `scale_x=False` can then prevent the terminal Pi-PLS fit from applying
  another predictor standardization while `scale_y=True` retains response standardization.
- Keep response-standardized CV-MSE semantics independent from model response scaling. The scorer
  continues to learn its normalization from each fitted training fold even when `scale_y=False`.
- Do not add block labels, block norms, block methods, or response-transformer semantics to
  `PiPLSRegression`. Those belong to a separate block-scaling component or a later dedicated design.

## Consequences

- Existing calls that use only `scale=True` or `scale=False` retain their previous numerical
  behavior.
- Predictor and response standardization can now be selected independently.
- Predictor block-scaling transformers can compose with `PiPLSSearchCV` through the already-supported
  pipeline boundary without forcing response scaling off.
- Ordinary scikit-learn `Pipeline` still transforms predictors rather than responses; this decision
  does not define a general response-transformer composition API.
