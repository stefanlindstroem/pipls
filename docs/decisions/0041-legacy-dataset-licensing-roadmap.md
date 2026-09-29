# Decision 0041: legacy dataset licensing roadmap

## Status

Accepted after review of the remaining companion-analysis datasets on 2026-07-19.

## Context

The companion analysis used Corn, Steel, SARCOS, and FRED-MD in addition to the repository's
current Pulp, Sugarcane, and Tobacco examples. Earlier planning treated those four datasets as
possible future integrations. The project owner has now set a stricter rule: `pipls` does not
redistribute a dataset when the licensing of the exact source material is doubtful.

Public accessibility is not itself a redistribution license. The review therefore distinguishes
between the exact companion-analysis source and later or third-party derivatives that may carry a
separate license.

### Corn

The publicly distributed corn spectroscopy material does not provide a sufficiently explicit
source-level grant for redistribution by this repository. Corn also overlaps substantially with the
existing Sugarcane and Tobacco NIR examples. It is intentionally excluded.

### Legacy Steel table

The companion materials identify the 267-row table as Citrination-processed. The matching public
source candidate is Citrination dataset 153092, *Mechanical properties of some steels*. Its page
describes more than 800 steels and identifies the contributor, but neither the exact derivation of
the companion table nor an explicit dataset license is documented there. The Open Citrination terms
restrict the service to academic, nonprofit, or noncommercial use and do not grant a general right
to copy and distribute hosted data.

Separately published Steel Strength derivatives contain 312 records and carry explicit permissive
licenses, but they are not the same 267-row, 13-predictor, four-property source table. Their licenses
do not establish redistribution rights for the companion table. The legacy Steel dataset is
therefore excluded. A future proposal may evaluate an explicitly licensed Steel derivative as a new
dataset with its own scientific purpose.

### SARCOS

The canonical Gaussian Process for Machine Learning page provides the official 44,484-row training
file and 4,449-row test file, describes the 21 inputs and seven torques, and credits Sethu
Vijayakumar. It does not state a dataset license or an explicit redistribution grant. The files are
therefore not included or repackaged by `pipls`.

### FRED-MD

The Federal Reserve Bank of St. Louis makes FRED-MD publicly accessible for macroeconomic research.
Current FRED terms nevertheless reserve rights in FRED content, restrict incorporation into other
databases, and require users to resolve the rights of third-party series owners. The historical
companion table also lacks a repository-facing record of its vintage, source-series rights, lag and
lead construction, and response deletion. `pipls` cannot redistribute that derived table with the
required confidence, so FRED-MD is excluded.

## Reviewed sources

- Citrination dataset 153092, *Mechanical properties of some steels*:
  `https://citrination.com/datasets/153092/show_files/`
- Open Citrination terms of service:
  `https://citrination.com/terms/`
- The separately licensed 312-record Steel Strength derivative:
  `https://doi.org/10.6084/m9.figshare.7250453`
- The Materials Data Facility publication of the Steel Strength derivative:
  `https://doi.org/10.18126/524z-vd6m`
- Gaussian Process for Machine Learning data page hosting SARCOS:
  `https://gaussianprocess.org/gpml/data/`
- Federal Reserve Bank of St. Louis FRED-MD and FRED-QD page:
  `https://www.stlouisfed.org/research/economists/mccracken/fred-databases`
- FRED terms of use and copyright statement:
  `https://fred.stlouisfed.org/legal/`

## Decision

- Retain Pulp, Sugarcane, and Tobacco as the complete current repository real-data suite.
- Do not add Corn, the legacy Citrination Steel table, SARCOS, or FRED-MD to `datasets/`, examples,
  source distributions, tests, or generated repository artifacts.
- Do not treat automatic downloading or reconstruction as a workaround for absent redistribution
  rights.
- Require every future dataset proposal to identify the exact source material and an explicit
  license or written permission granting redistribution and adaptation for general use before
  implementation starts.
- Treat a differently licensed derivative as a distinct candidate dataset. Its license does not
  clear an upstream or sibling dataset, and it must independently add a useful package-level case.
- Permit downstream paper-reproduction repositories to document user-obtained external data under
  the applicable source terms, without making those files part of `pipls`.
- Resume product documentation and release hardening; no legacy dataset remains pending.

## Consequences

- The repository carries only datasets whose redistribution basis is clear and reviewable from the
  committed materials.
- The current real-data suite remains intentionally small rather than mirroring every companion
  analysis.
- Earlier Corn-integration planning is superseded.
- Steel, SARCOS, and FRED-MD may be reconsidered only after a new owner decision based on an exact,
  explicitly licensed source and a distinct package-level purpose.
- Dataset licensing is resolved before data preparation, examples, or tests are designed.
