# Decision 0093: producer-owned immutable core public results

## Status

Accepted. Revised to remove exhaustive direct-construction validation from returned result records.

## Context

The fixed-estimator and search layers return frozen records such as `PiPLSDecomposition`,
`PiPLSSelection`, `PiPLSPredictorRankProfile`, `PiPLSComponentPath`, and `PiPLSOOFReport`.
These are returned-first interfaces: their semantic contents are produced by fitted estimators,
search logic, and OOF computation rather than supplied independently by users.

An earlier implementation nevertheless treated each dataclass constructor as a second complete
validation boundary. That duplicated invariants already enforced by the producing algorithms,
required large cross-field `__post_init__` methods, normalized scalar types solely for direct
construction, and generated many tests for malformed states that normal package workflows cannot
produce.

## Decision

1. Estimators, search methods, and report-producing functions own semantic validity. They must not
   emit inconsistent component counts, ranks, scores, provenance, OOF coverage, or numerical
   diagnostics.
2. Result dataclass constructors are not general-purpose validators for arbitrary user-created
   states. Malformed direct construction is outside the supported contract.
3. Array-containing result records still make defensive copies in their documented dtypes and keep
   those arrays read-only. This storage boundary prevents aliasing of producer-owned results.
4. Array-containing records reconstruct through their constructors when unpickled so NumPy arrays
   regain the read-only storage contract. Scalar-only records use ordinary dataclass pickling.
5. Scalar-only records do not normalize NumPy scalars or revalidate cross-field provenance merely
   for direct construction. Package producers emit the documented Python scalar types.
6. Public user inputs remain validated at the estimator, search, and inspection entry points that
   consume them. Removing duplicate result-constructor validation does not relax those interfaces.

## Consequences

Valid estimator-, search-, and report-produced values, field names, immutability, and serialized
array behavior are unchanged. The result layer becomes substantially smaller and tests focus on
producer behavior plus defensive storage rather than on impossible malformed constructor states.
Code that directly constructs inconsistent public result dataclasses no longer has guaranteed
validation behavior.
