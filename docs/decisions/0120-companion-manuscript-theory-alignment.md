# Decision 0120: companion-manuscript theory alignment

## Status

Accepted and implemented for companion-manuscript theory alignment. Decision 0155 leaves this
publication contract intact while authorizing a software-only least-squares response-subspace
extension.

## Context

The public theory page described the implemented fixed construction correctly at a high level, but
it was substantially shorter than the companion manuscript's derivation. It omitted the predictor
projector decomposition, the response-subspace optimization, the diagonal latent regression
relation, the meaning of “panoramic,” and several important limiting-method relationships. Its
“Reference and scope” section also cited the Pulp application paper, which documents dataset
provenance and application context rather than the theory of Pi-PLS.

The repository-maintained theory contract also used “predictor signal subspace” too generally and
counted the entries of $P$ as independently free when describing model dimension. In the companion
manuscript, once $\Pi$ is fixed, $P=\Pi M$ is constrained to the retained $r_\pi$-dimensional
subspace. Section 3.3 therefore derives the nominal fitted dimension as
$(r_\pi+q-h)h$. A later discussion paragraph uses an incomplete shorthand; the explicit derivation
is the internally consistent source.

## Decision

Make the companion manuscript the explicit scientific reference for the canonical theory page and
replace the former reference-and-scope material entirely.

The canonical derivation must include:

1. the centered multivariate regression setting;
2. the retained predictor basis $\Pi$ and projector $\Pi\Pi^{\mathsf T}$;
3. the predictor decomposition and truncation residual;
4. the retained scores $Z=X\Pi$;
5. the response-subspace optimization
   \[
   \max_{C^{\mathsf T}C=I_h}\|Z^{\mathsf T}YC\|_F^2;
   \]
6. the latent least-squares map $W=Z^+YC$;
7. the SVD $W=MDN^{\mathsf T}$;
8. $P=\Pi M$, $Q=CN$, and the diagonal relation
   \[
   YQ=XPD+E_\pi;
   \]
9. the equivalent regression maps
   \[
   PDQ^{\mathsf T}=\Pi WC^{\mathsf T};
   \]
10. the OLS, RRR, CCA, PLS, and PLS-SVD relationships;
11. the panoramic interpretation as coupling modes obtained from one undeflated retained predictor
    representation;
12. the nominal fitted dimension $(r_\pi+q-h)h$ after $\Pi$ is fixed.

Use “retained predictor subspace” for observed data. Reserve “predictor signal rank” for synthetic
settings where the noiseless generating rank is known. State explicitly that orthogonal residual
rotation preserves the Frobenius norm and covariance eigenvalues, not generally the covariance
matrix itself.

Keep the package boundary explicit. Estimator preprocessing, numerical-rank checks, search policies,
validation protocols, and practical real-data workflows are software capabilities around the fixed
construction and need not duplicate manuscript experimental choices. The Pulp source paper remains
cited in dataset and tutorial documentation, not as the theory reference.

This decision changes documentation contracts only. It does not change package source behavior,
defaults, examples, benchmarks, model selection, validation, or real-data workflows.

## Relationship to Decision 0155

The response-subspace optimization in item 5 remains the canonical construction documented by the
peer-reviewed companion manuscript and remains the package default. Decision 0155 authorizes an
additional least-squares-driven response-subspace policy as a software extension; it does not alter
what the companion manuscript defines or retrospectively place that extension in the publication.

Theory documentation now presents the software-only alternative alongside this canonical
derivation and labels the publication boundary explicitly. Manuscript-reproduction workflows
continue to use the cross-covariance construction.

## Consequences

- The public theory page follows the companion manuscript's fixed mathematical construction.
- The canonical repository equations now include the omitted optimization and projector identities.
- The model-dimension contract uses the manuscript's explicit Section 3.3 derivation.
- Observed predictor SVD directions are no longer described as guaranteed signal directions.
- Package behavior and practical workflows remain unchanged.
- A later terminology decision may still refine vocabulary across the broader public documentation
  and docstring surface.
