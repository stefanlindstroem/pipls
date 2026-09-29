# Reference datasets

Π-PLS distributes three reviewed real-data regression datasets as installed package resources:
[Pulp](#pulp-real-data-integration), [Sugarcane](#sugarcane-spectral-integration), and
[Tobacco](#tobacco-spectral-integration). This page owns their provenance, licensing, adaptation,
matrix dimensions, and direct resource locations. Exact loader and container signatures are in the
[Datasets and generators API](api/datasets.md).

Real-data reading remains user-owned in general. Π-PLS provides no generic registry, downloader, or
required metadata sidecar for user data. The three named reference datasets are explicit exceptions:
they are bundled with the installed distribution and require no network access.

```python
from pipls.datasets import load_pulp, load_sugarcane, load_tobacco

pulp = load_pulp()
sugarcane = load_sugarcane()
tobacco = load_tobacco()
X, Y = load_tobacco(return_X_y=True)
```

The default result is a `PiPLSDataset` containing read-only model arrays, names, and metadata. The
scikit-learn-style `return_X_y=True` mode returns the same read-only `float64` matrices directly.
The loaders apply no imputation, centering, scaling, row filtering, or learned preprocessing.

## Using the raw files outside Python

The named loaders read ordinary language-neutral resources. The same files can be consumed directly
from R, C++, MATLAB, Julia, or another environment. For reproducible non-Python use, prefer a tagged
source release, source distribution, or wheel rather than a moving development branch or an
installation path tied to one environment.

In a source checkout or unpacked source distribution, each dataset uses the same layout:

```text
src/pipls/_data/<dataset>/
├── X.csv
├── Y.csv
├── metadata.json
├── README.md
└── LICENSE.txt
```

Here `<dataset>` is `pulp`, `sugarcane`, or `tobacco`. A wheel is a ZIP archive containing the same
layout below `pipls/_data/<dataset>/`; an installed environment normally places it below
`<site-packages>/pipls/_data/<dataset>/`. Each `X.csv` and `Y.csv` pair is exactly the matrix pair
returned by its Python loader; loading applies no additional preprocessing.

## Pulp real-data integration {#pulp-real-data-integration}

The installed Pulp resources contain 46 rows, 14 fiber-property predictors, and eight responses
adapted from the supplementary material for the following article:

> Lindström, S. B., Ferritsius, R., Carlson, J. E., Persson, J., and Nilsson, F. (2025).
> Predicting handsheet properties and enhancing refiner control using fiber analyzer data and
> latent variable modeling. *Computers & Chemical Engineering*, **199**, 109143.
> [doi:10.1016/j.compchemeng.2025.109143](https://doi.org/10.1016/j.compchemeng.2025.109143).

The article identifies the refiner controls, internal state variables, pulp descriptions, and
handsheet properties as supplementary data. The package-owned dataset selects the documented
fiber-property and response columns from that public supplementary table.

`load_pulp()` reads the installed package resources and returns labels and metadata together with
the two model matrices. Public provenance is stored once under `dataset.metadata["provenance"]`.
The installed resources are the sole active Pulp matrix representation.

## Sugarcane spectral integration {#sugarcane-spectral-integration}

The installed Sugarcane resources contain 57 samples, 1,721 LabSpec absorbance predictors spanning
780 through 2500 nm, and four responses: total sugar, crude protein, acid detergent fiber, and in
vitro
organic matter digestibility. The original public dataset is:

> Chaix, G., Bendoula, R., and Zgouz, A. (2021). Data set of Visible-Near Infrared handled and
> micro-spectrometers -- comparison of their accuracy for predicting some sugarcane properties.
> *Mendeley Data*, Version 1.
> [doi:10.17632/mjttsjfj2s.1](https://doi.org/10.17632/mjttsjfj2s.1).

The accompanying data paper is:

> Zgouz, A., Héran, D., Barthès, B., Bastianelli, D., Bonnal, L., Baeten, V., Lurol, S.,
> Bonin, M., Roger, J.-M., Bendoula, R., and Chaix, G. (2020). Dataset of visible-near infrared
> handheld and micro-spectrometers -- comparison of the prediction accuracy of sugarcane
> properties. *Data in Brief*, **31**, 106013.
> [doi:10.1016/j.dib.2020.106013](https://doi.org/10.1016/j.dib.2020.106013).

The Mendeley collection is licensed CC BY 4.0. The package adaptation matches the public LabSpec
and response tables by `Sample`, removes three rows whose total-sugar response is missing, and
applies no imputation or spectral preprocessing. `load_sugarcane()` returns the labeled package
dataset with read-only model arrays or those arrays directly. The compact spectral-axis description
in `metadata.json` avoids repeating 1,721 equivalent per-wavelength descriptions while still
defining every predictor column exactly.

## Tobacco spectral integration {#tobacco-spectral-integration}

The installed Tobacco resources contain 347 samples, 1,557 raw FT-NIR absorbance predictors
spanning approximately 10,001 through 4,000 $\mathrm{cm}^{-1}$, and 13 chemical-component responses. The
original public dataset is:

> Chen, H., Guo, J., Wang, H., and Zhao, L. (2025). A Near-Infrared Spectroscopy Dataset for
> Chemical Composition Prediction and Origin Identification of Tobacco Leaves. *Mendeley Data*,
> Version 1. [doi:10.17632/9z7dgdtggk.1](https://doi.org/10.17632/9z7dgdtggk.1).

The related data paper is:

> Chen, H., Guo, J., Li, B., Wan, R., Wang, C., Su, M., Wang, X., Liu, R., Wang, S., Liu, K.,
> Chen, L., Yang, S., Xie, F., Nie, C., Zhao, L., Wang, H., and Liu, Z. (2026). A dataset for
> geographical origin identification of tobacco leaves from multiple countries using near-infrared
> spectroscopy and chemometric analysis. *Data in Brief*, **64**, 112418.
> [doi:10.1016/j.dib.2025.112418](https://doi.org/10.1016/j.dib.2025.112418).

The Mendeley collection is licensed CC BY 4.0. The package adaptation matches the public spectra
and chemistry tables one-to-one by sample ID, orders rows by that identifier, and excludes only
source metadata columns from the model matrices. All samples and chemical responses are retained.
No imputation, smoothing, derivatives, scatter correction, centering, scaling, or other spectral
preprocessing is applied. `load_tobacco()` returns the labeled package dataset with read-only model
arrays or those arrays directly; metadata retains the source-order response names and spectral-axis
description.
