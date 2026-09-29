# Decision 0117: commercial license, authorship, and citation

## Status

Accepted and implemented.

## Context

The repository already declared `BSD-3-Clause` in package metadata, but the root license named only
"Pi-PLS authors" and contained an abbreviated disclaimer rather than the complete standard text.
The public documentation did not clearly distinguish the software and documentation authors from
the three authors of the companion article or provide a stable canonical citation for the method
and package.

The owner requires commercial and noncommercial use, redistribution, and modification to remain
permitted while the copyright notice, conditions, and disclaimer must be retained. The package also
contains separately licensed reference datasets, so the repository license must not be described as
relicensing those assets.

The `pipls` code and repository-authored documentation are written by Vishal Agrawal and Stefan B.
Lindström. The companion article is published in *Computers & Chemical Engineering* and additionally
includes Fritjof Nilsson as a co-author. The article is the canonical citation for PiPLS.

## Decision

Retain the BSD 3-Clause License. Replace the abbreviated root text with the complete standard
BSD-3-Clause terms and name Vishal Agrawal and Stefan B. Lindström as the 2026 copyright holders of
the repository-authored code and documentation.

Use those two names for software authorship in package metadata and public documentation. Retain
Vishal Agrawal, Fritjof Nilsson, and Stefan B. Lindström as the authors of the companion article.
Maintain `docs/citation.md` as the human-readable ownership, license-scope, and companion-article
page. Do not prescribe a separate software citation; direct users to the companion article as the
single canonical citation.

Retain the root `CITATION.cff` using Citation File Format 1.2.0 because GitHub and compatible tools
consume it as machine-readable citation metadata. Its top-level fields continue to identify the
software artifact and release, while `preferred-citation` and its user-facing message direct users
to the companion article rather than requesting a separate software citation.

The companion-paper reference is:

> Agrawal, V., Nilsson, F., and Lindström, S. B. (2026). Panoramic Partial Least Squares
> (Π-PLS): A transparent and parsimonious multivariate regression model with paired latent
> directions. *Computers & Chemical Engineering*, 109913.
> https://doi.org/10.1016/j.compchemeng.2026.109913

Dataset-specific license and attribution files remain authoritative for the reference datasets.
The BSD-3-Clause repository license does not relicense those datasets. This decision does not add
paper-reproduction assets, claim publication or acceptance, or authorize package release work.

## Consequences

- Commercial use, redistribution, and modification remain permitted under BSD-3-Clause.
- Source and binary redistributors must preserve the notice, conditions, and disclaimer as stated
  in the license.
- The software and documentation authors and current copyright holders are distinct from the
  companion-article author list.
- Human and machine users are directed to the companion article through one maintained citation
  route.
- Dataset-specific licenses remain visible and legally separate.
- Publication metadata will require a focused update after the companion paper is published.
