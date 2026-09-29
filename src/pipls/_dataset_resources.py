"""Packaged reference-dataset loading."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from importlib import resources
from typing import Literal, cast, overload

import numpy as np

from ._dataset_types import FloatArray, PiPLSDataset


@dataclass(frozen=True)
class _PackagedDatasetConfig:
    dataset_id: str
    display_name: str


_PULP_DATASET = _PackagedDatasetConfig(
    dataset_id="pulp",
    display_name="Pulp",
)
_SUGARCANE_DATASET = _PackagedDatasetConfig(
    dataset_id="sugarcane",
    display_name="Sugarcane",
)
_TOBACCO_DATASET = _PackagedDatasetConfig(
    dataset_id="tobacco",
    display_name="Tobacco",
)


@overload
def load_pulp(*, return_X_y: Literal[False] = False) -> PiPLSDataset:
    ...


@overload
def load_pulp(*, return_X_y: Literal[True]) -> tuple[FloatArray, FloatArray]:
    ...


def load_pulp(
    *,
    return_X_y: bool = False,
) -> PiPLSDataset | tuple[FloatArray, FloatArray]:
    """Load the packaged Pulp fiber-property regression dataset.

    The dataset contains 46 thermomechanical-pulp samples, 14 fiber-property
    predictors, and eight pulp or handsheet responses. The analysis-facing
    matrices preserve the numeric values, column order, and row order selected
    from the publication supplementary material. No preprocessing is applied.

    Parameters
    ----------
    return_X_y : bool, default=False
        If ``True``, return the read-only predictor and response arrays directly.
        Otherwise return a :class:`PiPLSDataset` with read-only arrays, labels,
        and metadata.

    Returns
    -------
    PiPLSDataset or tuple of ndarray
        Structured dataset by default, or ``(X, Y)`` when ``return_X_y=True``.

    Notes
    -----
    The dataset is adapted from supplementary material for Lindström et al.
    (2025), *Computers & Chemical Engineering*, 199, 109143,
    doi:10.1016/j.compchemeng.2025.109143, under CC BY 4.0.
    """

    return _load_packaged_dataset(
        _PULP_DATASET,
        return_X_y=return_X_y,
    )


@overload
def load_sugarcane(*, return_X_y: Literal[False] = False) -> PiPLSDataset:
    ...


@overload
def load_sugarcane(*, return_X_y: Literal[True]) -> tuple[FloatArray, FloatArray]:
    ...


def load_sugarcane(
    *,
    return_X_y: bool = False,
) -> PiPLSDataset | tuple[FloatArray, FloatArray]:
    """Load the packaged Sugarcane LabSpec regression dataset.

    The dataset contains 57 sugarcane samples, 1,721 visible-near-infrared
    absorbance predictors at integer wavelengths from 780 through 2500 nm,
    and four chemical or feed-quality responses. The matrices preserve the
    retained numeric values, column order, and row order derived from the
    public source tables. No preprocessing is applied during loading.

    Parameters
    ----------
    return_X_y : bool, default=False
        If ``True``, return the read-only predictor and response arrays directly.
        Otherwise return a :class:`PiPLSDataset` with read-only arrays, labels,
        and metadata.

    Returns
    -------
    PiPLSDataset or tuple of ndarray
        Structured dataset by default, or ``(X, Y)`` when ``return_X_y=True``.

    Notes
    -----
    The dataset is adapted from Chaix, Bendoula, and Zgouz (2021), Mendeley
    Data, Version 1, doi:10.17632/mjttsjfj2s.1, under CC BY 4.0. Source samples
    103, 105, and 111 are excluded because total sugar is missing.
    """

    return _load_packaged_dataset(
        _SUGARCANE_DATASET,
        return_X_y=return_X_y,
    )


@overload
def load_tobacco(*, return_X_y: Literal[False] = False) -> PiPLSDataset:
    ...


@overload
def load_tobacco(*, return_X_y: Literal[True]) -> tuple[FloatArray, FloatArray]:
    ...


def load_tobacco(
    *,
    return_X_y: bool = False,
) -> PiPLSDataset | tuple[FloatArray, FloatArray]:
    """Load the packaged Tobacco leaf FT-NIR regression dataset.

    The dataset contains 347 tobacco leaf samples, 1,557 raw FT-NIR
    absorbance predictors ordered from approximately 10,001 down to
    4,000 $\\mathrm{cm}^{-1}$, and 13 chemical-component responses. The matrices
    preserve the retained numeric values, column order, and row order
    derived from the public source workbooks. No preprocessing is applied.

    Parameters
    ----------
    return_X_y : bool, default=False
        If ``True``, return the read-only predictor and response arrays directly.
        Otherwise return a :class:`PiPLSDataset` with read-only arrays, labels,
        and metadata.

    Returns
    -------
    PiPLSDataset or tuple of ndarray
        Structured dataset by default, or ``(X, Y)`` when ``return_X_y=True``.

    Notes
    -----
    The dataset is adapted from Chen, Guo, Wang, and Zhao (2025),
    Mendeley Data, Version 1, doi:10.17632/9z7dgdtggk.1, under CC BY 4.0.
    """

    return _load_packaged_dataset(
        _TOBACCO_DATASET,
        return_X_y=return_X_y,
    )


def _load_packaged_dataset(
    config: _PackagedDatasetConfig,
    *,
    return_X_y: bool,
) -> PiPLSDataset | tuple[FloatArray, FloatArray]:
    if not isinstance(return_X_y, bool):
        raise TypeError("return_X_y must be a boolean.")

    metadata = _load_dataset_metadata(config)
    feature_names = tuple(cast(list[str], metadata["feature_names"]))
    target_names = tuple(cast(list[str], metadata["target_names"]))
    X = _load_dataset_csv(config, "X.csv")
    Y = _load_dataset_csv(config, "Y.csv")

    dataset = PiPLSDataset(
        X=X,
        Y=Y,
        feature_names=feature_names,
        target_names=target_names,
        metadata=metadata,
    )
    if return_X_y:
        return dataset.X, dataset.Y
    return dataset


def _dataset_resource_bytes(config: _PackagedDatasetConfig, name: str) -> bytes:
    resource = (
        resources.files("pipls")
        .joinpath("_data")
        .joinpath(config.dataset_id)
        .joinpath(name)
    )
    try:
        return resource.read_bytes()
    except (FileNotFoundError, OSError) as error:
        message = f"Packaged {config.display_name} resource {name!r} is unavailable."
        raise RuntimeError(message) from error


def _load_dataset_metadata(config: _PackagedDatasetConfig) -> dict[str, object]:
    try:
        loaded = json.loads(
            _dataset_resource_bytes(config, "metadata.json").decode("utf-8")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Packaged {config.display_name} metadata is invalid.") from error
    return cast(dict[str, object], loaded)


def _load_dataset_csv(
    config: _PackagedDatasetConfig,
    name: str,
) -> FloatArray:
    raw = _dataset_resource_bytes(config, name)

    try:
        rows = csv.reader(io.StringIO(raw.decode("utf-8"), newline=""))
        next(rows)
        values = [[float(value) for value in row] for row in rows]
        return cast(FloatArray, np.asarray(values, dtype=np.float64))
    except (StopIteration, UnicodeDecodeError, ValueError) as error:
        message = (
            f"Packaged {config.display_name} resource {name!r} "
            "is not a valid numeric CSV."
        )
        raise RuntimeError(message) from error
