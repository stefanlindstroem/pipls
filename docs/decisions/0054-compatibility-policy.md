# Decision 0054: compatibility policy

## Status

Accepted and implemented.

## Context

The package declared lower bounds for Python, NumPy, scikit-learn, and joblib, but it did not define
a complete supported interpreter range, protect users from unreviewed dependency major versions,
or test one reproducible minimum-dependency combination. The ordinary CI matrix stopped at Python
3.13, and its minimum-version job constrained only scikit-learn.

The oldest supported NumPy and scikit-learn lines do not publish wheels for every newer Python
version. A complete Cartesian product of interpreter and dependency versions is therefore neither
possible nor necessary.

## Decision

- Support Python 3.10 through 3.14.
- Keep `requires-python = ">=3.10"` without an upper bound, while using project classifiers and
  documentation to enumerate the supported versions.
- Declare the runtime ranges `numpy>=1.26,<3`, `scikit-learn>=1.4,<2`, and `joblib>=1.2,<2`.
- Treat the upper major-version limits as compatibility guards that require explicit review before
  a new dependency major is admitted.
- Record the minimum tested dependency lines in `constraints/minimum.txt`: NumPy 1.26,
  scikit-learn 1.4, and joblib 1.2.
- Test that minimum combination on Python 3.10.
- Test Python 3.10 through 3.14 with normal dependency resolution rather than attempting every
  interpreter/dependency combination.
- Treat the constraint file as a maintainer test input, not as an application lock file or an
  installation recommendation for users.
- Keep optional dependency ranges separate from the core runtime compatibility claim.
- Exercise three explicit CI responsibilities: the Python 3.10 minimum stack, normal dependency
  resolution on every supported Python version, and latest-compatible upgrades on Python 3.14.
- Print resolved Python and runtime dependency versions in compatibility jobs.
- Build release artifacts once, then validate clean wheel and source-distribution installations in
  isolated environments with the same public smoke contract.

## Consequences

Package metadata, documentation, minimum constraints, and the existing CI boundary now describe one
coherent support policy. The lower bounds are executable rather than aspirational, and Python 3.14
is included in the supported matrix. New compatible minor dependency releases remain installable,
while unreviewed major releases are excluded until the project deliberately expands its range.

The compatibility matrix remains intentionally small and separates minimum, supported-Python, and
latest-compatible responsibilities. Clean installed wheel and source-distribution validation is
part of the same release-compatibility boundary rather than an editable-checkout substitute.
