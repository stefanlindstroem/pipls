# Decision 0110: response-anchored Pi-PLS display factors

## Status

Accepted.

## Context

`pipls_display_factors()` previously resolved the sign indeterminacy of every Pi-PLS component only
through the first largest-magnitude predictor-direction entry. That deterministic convention is a
suitable general default, but some applications have one principal controlled response whose
orientation is more useful for interpreting all paired components. The Pulp analysis uses tensile
index (`TI`) in this role.

The fitted decomposition intentionally contains numerical arrays rather than scientific labels, and
`PiPLSDisplayFactors` intentionally contains factor values rather than sign-selection bookkeeping.
Any extension must preserve those boundaries and must not change the fitted regression map.

## Decision

`pipls_display_factors()` retains its existing predictor-anchored behavior when called with only a
`PiPLSDecomposition`. It adds keyword-only `response_index` and `response_sign` parameters:

```python
pipls_display_factors(
    decomposition,
    response_index=None,
    response_sign="positive",
)
```

`response_index` is either `None` or a nonnegative zero-based response row. When it is supplied,
`response_sign="positive"` orients every component so the selected response-direction entry is
nonnegative, while `response_sign="negative"` makes it nonpositive. An exactly zero selected entry
cannot define an orientation and therefore falls back to the existing predictor-based convention for
that component. A negative response sign without a response index is rejected rather than ignored.

The same component sign is applied to the paired columns of $P$ and $Q$; $D$ is unchanged and $QD$
is recomputed from the signed $Q$. Therefore

\begin{equation}
P_{\mathrm{display}}D Q_{\mathrm{display}}^{\mathsf T}=PDQ^{\mathsf T}.
\end{equation}

The helper accepts an index rather than a response name because labels remain caller-owned. The Pulp
example and tutorial resolve `response_names.index("TI")` explicitly and request a positive
orientation. The result record gains no sign-convention or anchor fields.

## Consequences

The default call remains backward compatible and deterministic. Applications can choose a
scientifically meaningful response orientation without mutating the fitted estimator or exposing
construction internals. The requested sign is a display convention, not a statement that every
physical effect on the anchored response has that sign. Tests cover both orientations, zero-entry
fallback, validation, regression-map preservation, and the TI-positive Pulp workflow.
