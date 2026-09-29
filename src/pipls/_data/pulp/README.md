# Pulp fiber-property regression data

These package resources support `pipls.datasets.load_pulp()`.

- `X.csv` contains 14 fiber-description predictors.
- `Y.csv` contains Canadian Standard Freeness and seven handsheet-property responses.
- `metadata.json` records labels, public provenance, preparation, dimensions, and integrity hashes.
- `LICENSE.txt` contains the attribution and CC BY 4.0 notice for this adapted dataset.

## Raw-file access

These files are intentionally usable independently of Python. In a source checkout or unpacked
source distribution, this directory is `src/pipls/_data/pulp/`. In a wheel, which is a ZIP
archive, it is `pipls/_data/pulp/`. In an installed environment it normally appears below
`<site-packages>/pipls/_data/pulp/`. Prefer a tagged source release, source distribution,
or wheel when a reproducible external-data input is required.

The public provenance reference is:

> Lindström, S. B., Ferritsius, R., Carlson, J. E., Persson, J., and Nilsson, F. (2025).
> Predicting handsheet properties and enhancing refiner control using fiber analyzer data and
> latent variable modeling. *Computers & Chemical Engineering*, **199**, 109143.
> [doi:10.1016/j.compchemeng.2025.109143](https://doi.org/10.1016/j.compchemeng.2025.109143).

The package adaptation selects the named fiber-property variables as predictors and the named pulp
and handsheet properties as responses. Refiner-control variables, specific refining energy, and the
final `k` variable are not part of the model matrices. All 46 rows, selected numeric values, column
order, and row order are preserved. No imputation, centering, scaling, row filtering, or learned
preprocessing is applied by the loader.
