# Authors, license, and citation

## Authors and copyright

The `pipls` code and repository-authored documentation are written by Vishal Agrawal and
Stefan B. Lindström.

Copyright (c) 2026 Vishal Agrawal and Stefan B. Lindström.

## Software and documentation license

The code and repository-authored documentation are distributed under the BSD 3-Clause License.
This permissive license allows source and binary use, redistribution, and modification, including
commercial use, subject to its conditions. Source redistributions must retain the copyright notice,
license conditions, and disclaimer. Binary redistributions must reproduce them in the accompanying
documentation or other materials. The non-endorsement condition also remains in force.

The complete and authoritative terms are in the repository-root `LICENSE` file.

The repository-level BSD license does not replace the licenses of included third-party or adapted
datasets. Each reference dataset retains its own attribution and license notice; see the
[reference dataset guide](datasets.md).

## Companion article {#companion-paper}

The companion article is the canonical citation for PiPLS:

> Agrawal, V., Nilsson, F., and Lindström, S. B. (2026). Panoramic Partial Least Squares
> (Π-PLS): A transparent and parsimonious multivariate regression model with paired latent
> directions. *Computers & Chemical Engineering*, 109913.
> [https://doi.org/10.1016/j.compchemeng.2026.109913](https://doi.org/10.1016/j.compchemeng.2026.109913)

The [theory overview](theory.md) summarizes the fixed mathematical construction from the
companion article and distinguishes it from package-level preprocessing, search, and validation
capabilities. The [Datasets and generators API](api/datasets.md#synthetic-generator) defines the
Gaussian data-generating distribution used for package validation and testing.

The repository-root `CITATION.cff` provides this preferred citation in machine-readable form for
GitHub and compatible citation tools.
