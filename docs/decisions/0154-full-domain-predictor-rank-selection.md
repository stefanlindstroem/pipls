# Decision 0154: full-domain predictor-rank selection and explicit EPV policy

## Status

Accepted and implemented. This decision changes the predictor-rank search contract but does not
change the fixed Pi-PLS numerical construction. The implementation is staged so that the accepted
contract, runtime behavior, tests, examples, and public documentation can be reviewed separately.

## Context

`PiPLSSearchCV` currently mixes three distinct concepts when resolving predictor rank:

1. hard feasibility imposed by fold-local predictor dimension, centering, and numerical rank;
2. statistical-support guidance based on `samples_per_predictor_rank`;
3. candidate-coverage strategy through exhaustive or adaptive search.

In particular, the ordinary search ceiling currently contains the support term
$\lceil n/c\rceil$, where $n$ is the full supplied sample count and
$c=\texttt{samples_per_predictor_rank}$. This prevents both automatic and exhaustive search from
examining otherwise feasible predictor ranks above that heuristic ceiling.

The package needs a different separation of responsibilities. Ordinary automatic model selection
must compare all predictor ranks that are mathematically and numerically admissible, unless the user
explicitly restricts the rank domain. The events-per-variable-inspired rule used in the companion
manuscript remains useful, but it is a specific rank-selection policy rather than a general bound on
search.

## Decision

### Separate hard feasibility from statistical heuristics

Let $p_{\mathrm{min}}$ be the smallest predictor dimension reaching the terminal `PiPLSRegression` across
the materialized training folds, let $n_{\mathrm{train,min}}$ be the smallest training-fold size,
and let $r_{\mathrm{num,min}}$ be the minimum verified numerical predictor rank across those folds.
The hard predictor-rank ceiling is

\begin{equation}
r_{\pi,\mathrm{hard}}
=
\min\left[
p_{\mathrm{min}},
n_{\mathrm{train,min}}-1,
r_{\mathrm{num,min}}
\right].
\end{equation}

These terms are feasibility constraints. They are not statistical heuristics and remain mandatory
for every search policy.

If `max_predictor_rank` is an explicit positive integer, it is a user-imposed upper restriction and
the effective ceiling becomes

\begin{equation}
r_{\pi,\mathrm{max}}
=
\min\left[r_{\pi,\mathrm{hard}},\texttt{max\_predictor\_rank}\right].
\end{equation}

If `max_predictor_rank=None`, then $r_{\pi,\mathrm{max}}=r_{\pi,\mathrm{hard}}$. The public parameter no
longer accepts the rule sentinel; it is only an explicit integer cap or `None`.

The ordinary automatic rank domain for a component count $h$ is therefore

\begin{equation}
\mathcal{R}_h
=
\{h,h+1,\ldots,r_{\pi,\mathrm{max}}\},
\end{equation}

subject to any explicit user-supplied predictor-rank sequence.

### Make exhaustive full-domain search the default

`predictor_rank_values=None` means that predictor rank is optimized over the complete admissible
rank domain. `search_method="exhaustive"` becomes the default candidate-coverage policy and
evaluates every admissible rank for every evaluated component count.

The exact configured-score reference optimum is therefore computed from the full admissible
candidate set. With the default scorer this reference is the minimum mean CV-MSE over that set.
Decision 0148 remains responsible for the subsequent predictor-rank tolerance stage: the retained
rank may be the smallest rank within the configured public tolerance of the exact reference optimum.
The default machine-scale tolerance continues to serve numerical parsimony rather than candidate
pruning.

`search_method="adaptive"` remains available as an explicit computational approximation. It uses
the same admissible rank domain but may evaluate only a subset and therefore retains the limitation
in Decision 0007 that it cannot guarantee the exhaustive optimum for an arbitrary discrete CV
curve.

Explicit multi-rank sequences define the user-requested optimization domain, after ordinary
component/rank compatibility and hard-feasibility validation. A one-rank sequence is a fixed rank
policy and performs no predictor-rank optimization.

### Add an explicit EPV policy

`predictor_rank_values="epv"` selects one nominal predictor rank from the events-per-variable-
inspired rule

\begin{equation}
r_{\pi,\mathrm{epv,nominal}}
=
\min\left[p,\left\lceil\frac{n}{c}\right\rceil\right],
\end{equation}

where $n$ is the full number of observations supplied to `fit()` and
$c=\texttt{samples_per_predictor_rank}>0$. The full supplied $n$ defines the EPV sample-count
convention and matches the manuscript workflow for the final model.

The effective EPV rank is the nominal value clipped only by unavoidable search feasibility and any
explicit user maximum:

\begin{equation}
r_{\pi,\mathrm{epv}}
=
\min\left[r_{\pi,\mathrm{epv,nominal}},r_{\pi,\mathrm{max}}\right].
\end{equation}

EPV is a fixed predictor-rank policy. It does not use predictor-rank parsimony tolerances and does
not carry `PiPLSPredictorRankEvidence`, because no conditional predictor-rank optimization occurs.
The component-count domain must be resolved after the effective EPV rank so that every evaluated
$h$ satisfies $h\leq r_{\pi,\mathrm{epv}}$.

The default `samples_per_predictor_rank` value becomes `10.0`, corresponding to the manuscript's
ordinary EPV choice. A value of `5.0` remains a more permissive EPV choice. Values below 5 remain
legal under the EPV policy and emit one `PredictorRankSupportWarning` per top-level search fit.
Thus a user who deliberately wants the most permissive EPV endpoint can write

```python
PiPLSSearchCV(
    predictor_rank_values="epv",
    samples_per_predictor_rank=1.0,
)
```

and receives the low-support warning. Since $c=1$ gives the nominal rank $\min(p,n)$, hard
fold-local feasibility then determines the effective rank. No separate `"max"` predictor-rank
heuristic is provided.

Outside the EPV policy, `samples_per_predictor_rank` does not constrain candidate ranks. The stored
default value is inert. A nondefault value is invalid unless `predictor_rank_values="epv"`, so code
written under the former assumption that $c$ limits ordinary automatic search fails visibly instead
of being silently ignored.

### Remove the maximum-rank policy vocabulary

`predictor_rank_values="max"` is removed without a compatibility alias. The corresponding public
predictor-rank provenance value `"maximum"` is also removed. The public predictor-rank policies
become:

- `"optimized"` for `predictor_rank_values=None` and explicit multi-rank sequences;
- `"fixed"` for one-rank explicit sequences;
- `"epv"` for the EPV policy.

A fixed or EPV policy with the constructor-default `search_method="exhaustive"` is valid. Because
its complete candidate set contains one predictor rank per compatible component count,
`search_is_exhaustive_` is true. The search method is relevant only when a policy exposes multiple
rank candidates.

### Preserve the hierarchical selection contract

For every component count $h$, `PiPLSSearchCV` continues to choose predictor rank before building
`component_path_`. Component-count selection then acts only on that conditioned path. Decision 0148
defines optimized predictor-rank tolerance selection, Decision 0146 defines component-count
tolerances, and Decision 0143 owns selected-row lookup, exact selection handoff, OOF reporting, and
final refitting.

No part of this decision changes the Pi-PLS factorization, the fixed-estimator rank checks, fold-
local preprocessing, configured scoring, OOF semantics, or final refit lifecycle.

## Relationship to earlier decisions

This decision refines the following retained contracts:

- Decision 0007: exhaustive rather than adaptive coverage becomes the default, and the former
  `predictor_rank_values="max"` fixed policy is removed. The adaptive and exhaustive algorithms
  otherwise remain in force.
- Decision 0009: `PredictorRankSupportWarning` is attached to EPV use with $c<5$, not to
  `max_predictor_rank="rule"`; the rule sentinel is removed.
- Decision 0039: the fixed-estimator/search ownership boundary remains unchanged, but the standard
  search is no longer bounded by the $c=5$ support rule and adaptive coverage is no longer the
  default.
- Decision 0092: fold numerical-rank preflight remains mandatory, but the general ceiling drops
  the $\lceil n/c\rceil$ support term. EPV applies its nominal rule before clipping by the verified
  hard ceiling.
- Decision 0148: predictor-rank tolerances remain applicable to optimized policies, while the
  former `"max"`/`"maximum"` policy clauses are replaced by the fixed `"epv"` policy described
  here.

All unaffected clauses of those decisions remain current.

## Implementation sequence

The accepted migration is intentionally staged:

1. record this decision and synchronize the maintainer roadmap;
2. separate hard-feasibility and EPV helper calculations without changing runtime behavior;
3. implement the new public search defaults, policy vocabulary, and rank-domain semantics;
4. add focused regression tests protecting full-domain, exhaustive, EPV, warning, and tolerance
   behavior;
5. update maintained examples and regenerate tutorial evidence from execution;
6. rewrite user and maintainer documentation around domain restriction, coverage strategy, and EPV;
7. complete the stale-contract, installed-package, distribution, and release audit.

The seven-patch migration is complete. Runtime behavior, focused regression coverage, maintained
examples, generated tutorial evidence, user and maintainer documentation, release notes, and clean
installed-artifact smoke tests now implement and verify this contract. The final audit also confirms
that active guidance contains no remaining `"max"`/`"rule"` rank-policy shortcut or hidden EPV
ceiling; historical changelog and decision text may retain those retired spellings as history.

## Consequences

- Ordinary automatic Pi-PLS search no longer treats an EPV-style support rule as a hidden search
  bound.
- Exhaustive means complete coverage of the user-visible admissible rank domain.
- The default search can be more expensive, especially when $p$ and fold numerical rank are large;
  users control that cost explicitly through rank sequences, an integer `max_predictor_rank`,
  component-count restriction, `search_method="adaptive"`, parallelism, and SVD policy.
- EPV remains available as an explicit manuscript-aligned fixed-rank policy rather than a general
  heuristic ceiling.
- The most permissive EPV choice is expressed transparently as $c=1$ and is accompanied by the
  existing low-support warning instead of being hidden behind a `"max"` shortcut.
- The accepted API has no compatibility aliases for the removed pre-release `"max"` or `"rule"`
  spellings.
