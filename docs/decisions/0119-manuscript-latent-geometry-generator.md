# Decision 0119: synthetic-data generator

## Status

Accepted and implemented; canonical single-generator contract.

## Context

Synthetic data in the package exists to support deterministic tests, validation, examples, and
reproducible demonstrations. It does not define the Pi-PLS estimator. Maintaining multiple public
synthetic frameworks therefore adds API and validation machinery without contributing to the
method.

The companion manuscript already provides a sufficient latent construction with
predictor-specific, shared, and response-specific variation. That construction is also suitable for
the package's maintained synthetic tests and examples, including train/test analyses obtained by
splitting generated rows.

## Decision

`make_synthetic_data()` is the single public synthetic generator and implements

\[
X = \Lambda_p L_p + \Lambda_s L_{sp} + \varepsilon_X,
\qquad
Y = \Lambda_s L_{sr} + \Lambda_r L_r + \varepsilon_Y.
\]

The three latent-score matrices and four loading matrices contain independent standard-normal
entries. Predictor and response noise contain independent Gaussian entries with caller-specified
standard deviations. The generator applies no score centering, score standardization, loading
orthonormalization, latent-strength scaling, or observed-variable scaling.

A fixed validated unsigned 32-bit `random_state` and identical arguments reproduce the same `X` and
`Y` arrays for a given package implementation. The internal random-draw order is not public API;
exact realization-level reproduction across package versions therefore requires recording the
package version as well as the seed and generator arguments.

The function returns the generated `(X, Y)` arrays directly. Latent score, loading, signal, and
noise matrices are implementation details rather than public result objects.

No separate train/test generator, latent-strength controls, score-distribution controls,
orthonormal-loading construction, or observed-variable scaling belong to the public synthetic-data
surface. When train/test blocks are required, callers generate `X` and `Y` once and split rows explicitly.

## Consequences

- The package has one synthetic model and one generator.
- The same generator supports focused package validation and the companion manuscript's synthetic
  distribution.
- Synthetic support code remains small and clearly separate from the Pi-PLS estimator.
- Exact reproduction of a manuscript figure or table still requires its parameter grid, seeds,
  resampling protocol, and analysis settings.
