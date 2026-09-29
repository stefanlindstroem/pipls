# Sugarcane LabSpec regression data

These package resources support `pipls.datasets.load_sugarcane()` and may also be used directly
from R, C++, MATLAB, Julia, or another environment.

The matrices are adapted from:

> Chaix, G., Bendoula, R., and Zgouz, A. (2021). Data set of Visible-Near Infrared handled and
> micro-spectrometers -- comparison of their accuracy for predicting some sugarcane properties.
> *Mendeley Data*, Version 1. doi:10.17632/mjttsjfj2s.1.

The related data paper is Zgouz et al. (2020), doi:10.1016/j.dib.2020.106013.

- `X.csv` contains LabSpec absorbance values at integer wavelengths from 780 through 2500 nm.
- `Y.csv` contains total sugar (`TS`), crude protein (`CP`), acid detergent fiber (`ADF`), and in
  vitro organic matter digestibility (`IVOMD`).
- `metadata.json` records ordered labels, source-row alignment, preparation, provenance, and
  integrity hashes.
- `LICENSE.txt` records attribution and the CC BY 4.0 data license.

The public source tables were joined one-to-one by `Sample`. Source samples 103, 105, and 111 were
excluded because `TS` is missing. The retained matrices contain 57 aligned rows, 1,721 predictors,
and four responses. Package-local identifiers `sugarcane-01` through `sugarcane-57` describe this
retained row order; the original source identifiers remain in `metadata.json`.

Loading applies no imputation, smoothing, derivative, scatter correction, centering, scaling, or
other spectral preprocessing. The CSV matrices are exactly those returned by the Python loader.
## Raw-file access

These files are intentionally usable independently of Python. In a source checkout or unpacked
source distribution, this directory is `src/pipls/_data/sugarcane/`. In a wheel, which is a ZIP
archive, it is `pipls/_data/sugarcane/`. In an installed environment it normally appears below
`<site-packages>/pipls/_data/sugarcane/`. Prefer a tagged source release, source distribution,
or wheel when a reproducible external-data input is required.
