# Packaged Tobacco leaf FT-NIR dataset

These package resources support `pipls.datasets.load_tobacco()` and may also be used directly
outside Python. The CSV matrices are the exact arrays returned by the loader; loading applies no
imputation, smoothing, derivative, scatter correction, centering, scaling, or other spectral
preprocessing.

## Files

- `X.csv`: 347 rows and 1,557 raw FT-NIR absorbance predictors. Headers are wavenumbers in
  `cm^-1`, ordered from approximately 10,001 down to 4,000.
- `Y.csv`: the same 347 rows and 13 chemical response columns in source order.
- `metadata.json`: dimensions, ordered labels, public provenance, preparation, and integrity
  hashes verified by package qualification tests.
- `LICENSE.txt`: Tobacco-specific attribution and Creative Commons Attribution 4.0 notice.

In a wheel, the files are stored under `pipls/_data/tobacco/`. In a source checkout or unpacked
source distribution, they are under `src/pipls/_data/tobacco/`.

## Raw-file access

These files are intentionally usable independently of Python. In a source checkout or unpacked
source distribution, this directory is `src/pipls/_data/tobacco/`. In a wheel, which is a ZIP
archive, it is `pipls/_data/tobacco/`. In an installed environment it normally appears below
`<site-packages>/pipls/_data/tobacco/`. Prefer a tagged source release, source distribution,
or wheel when a reproducible external-data input is required.

## Public source

The original source is Chen, Guo, Wang, and Zhao (2025), *A Near-Infrared Spectroscopy Dataset for
Chemical Composition Prediction and Origin Identification of Tobacco Leaves*, Mendeley Data,
Version 1, doi:10.17632/9z7dgdtggk.1.

The related article is Chen et al. (2026), *A dataset for geographical origin identification of
tobacco leaves from multiple countries using near-infrared spectroscopy and chemometric analysis*,
*Data in Brief*, 64, 112418, doi:10.1016/j.dib.2025.112418.

The package adaptation matches the spectra and chemistry tables one-to-one by their shared sample
identifier, orders rows by that identifier, removes non-model metadata columns, and preserves all
347 samples, 1,557 spectral columns, and 13 response columns.
