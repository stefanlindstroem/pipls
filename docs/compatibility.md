# Compatibility

Each dependency has a minimum supported version and a conservative tested upper major-version
limit. An upper limit may be extended after compatibility testing.

Python 3.10 through 3.14 is supported. Package metadata keeps
`requires-python = ">=3.10"` without an upper bound.

| Dependency | Supported range |
|---|---|
| NumPy | `>=1.26,<3` |
| scikit-learn | `>=1.4,<2` |
| joblib | `>=1.2,<2` |

The minimum tested stack is recorded in `constraints/minimum.txt`:

```text
numpy==1.26.*
scikit-learn==1.4.*
joblib==1.2.*
```

Continuous integration tests the minimum stack on Python 3.10, all supported Python versions with
normal dependency resolution, and the latest admitted dependency versions on Python 3.14. Clean
wheel and source-distribution installations are verified by `make dist-check`. The
[reproducibility guide](reproducibility.md) describes that
isolated artifact check and the separate documentation-distribution qualification.
