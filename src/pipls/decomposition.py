"""Public immutable representation of a fitted Π-PLS factorization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ._core import PiPLSCoreResult, ResolvedSVDSolver
from ._result_validation import _read_only_float_array

__all__ = [
    "PiPLSDecomposition",
]

FloatArray = NDArray[np.float64]

@dataclass(frozen=True)
class PiPLSDecomposition:
    r"""Interpretable Π-PLS factorization and numerical diagnostics.

    Instances are returned through :attr:`pipls.PiPLSRegression.decomposition_`.
    Arrays are stored as defensive read-only copies. Semantic validity is owned
    by the fitted estimator that produces the record; internal construction
    matrices used before the final factorization are intentionally not exposed.

    Attributes
    ----------
    predictor_directions : ndarray of shape (n_features, n_components)
        Orthonormal predictor directions $\mathbf{P}$ of the centered/scaled regression
        map.
    dilation : ndarray of shape (n_components,)
        Nonnegative dilations $D_k=D_{kk}$ of the paired latent modes.
    response_directions : ndarray of shape (n_targets, n_components)
        Orthonormal response directions $\mathbf{Q}$.
    predictor_numerical_rank : int
        Complete numerical predictor rank under full SVD, or a verified lower
        bound under truncated randomized SVD.
    predictor_numerical_rank_is_exact : bool
        Whether ``predictor_numerical_rank`` is the complete numerical rank.
    rank_tolerance : float
        Nonnegative tolerance used to classify retained predictor singular values.
    predictor_svd_solver : {"full", "randomized"}
        Predictor SVD implementation actually used.
    """

    predictor_directions: FloatArray
    dilation: FloatArray
    response_directions: FloatArray
    predictor_numerical_rank: int
    predictor_numerical_rank_is_exact: bool
    rank_tolerance: float
    predictor_svd_solver: ResolvedSVDSolver

    def __post_init__(self) -> None:
        """Store defensive read-only copies of the factor arrays."""

        object.__setattr__(
            self,
            "predictor_directions",
            _read_only_float_array(
                self.predictor_directions,
                name="predictor_directions",
                ndim=2,
            ),
        )
        object.__setattr__(
            self,
            "dilation",
            _read_only_float_array(self.dilation, name="dilation"),
        )
        object.__setattr__(
            self,
            "response_directions",
            _read_only_float_array(
                self.response_directions,
                name="response_directions",
                ndim=2,
            ),
        )

    def __reduce__(self) -> tuple[type[PiPLSDecomposition], tuple[object, ...]]:
        """Reconstruct so unpickled arrays remain read-only."""

        return (
            type(self),
            (
                self.predictor_directions,
                self.dilation,
                self.response_directions,
                self.predictor_numerical_rank,
                self.predictor_numerical_rank_is_exact,
                self.rank_tolerance,
                self.predictor_svd_solver,
            ),
        )

    @classmethod
    def _from_core_result(cls, result: PiPLSCoreResult) -> PiPLSDecomposition:
        return cls(
            predictor_directions=result.P,
            dilation=np.diag(result.D),
            response_directions=result.Q,
            predictor_numerical_rank=result.x_rank,
            predictor_numerical_rank_is_exact=result.x_rank_is_exact,
            rank_tolerance=result.rank_tolerance,
            predictor_svd_solver=result.predictor_svd_solver,
        )

    @property
    def standardized_regression_map(self) -> FloatArray:
        r"""Return the centered/scaled map $\mathbf{P}\mathbf{D}\mathbf{Q}^{\mathsf T}$."""

        return _read_only_float_array(
            np.asarray(
                (self.predictor_directions * self.dilation[None, :])
                @ self.response_directions.T,
                dtype=np.float64,
            ),
            name="standardized_regression_map",
            ndim=2,
        )
