"""Theory-faithful fixed-parameter Π-PLS numerical core.

This private module operates on predictor and response matrices that have already
been centered and, when requested by a caller, scaled. It contains no estimator,
cross-validation, or preprocessing policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.utils.extmath import randomized_svd

FloatArray = NDArray[np.float64]
SVDSolver = Literal["full", "randomized", "auto"]
ResolvedSVDSolver = Literal["full", "randomized"]
ResponseSubspace = Literal["cross_covariance", "least_squares"]
_AUTO_RANDOMIZED_MIN_DIMENSION = 500
_AUTO_RANDOMIZED_MIN_ENTRIES = 1_000_000
_AUTO_RANDOMIZED_MAX_RANK_FRACTION = 0.2
_MAX_RANDOM_STATE = int(np.iinfo(np.uint32).max)


class _PredictorRankInfeasibleError(ValueError):
    """Report a requested rank above the predictor rank verified by the SVD."""

    def __init__(
        self,
        *,
        requested_rank: int,
        verified_rank: int,
        rank_is_exact: bool,
        tolerance: float,
    ) -> None:
        self.requested_rank = requested_rank
        self.verified_rank = verified_rank
        self.rank_is_exact = rank_is_exact
        self.tolerance = tolerance
        rank_description = (
            "numerical rank" if rank_is_exact else "verified retained rank"
        )
        super().__init__(
            f"predictor_rank exceeds the {rank_description} of X: "
            f"predictor_rank={requested_rank}, "
            f"{rank_description.replace(' ', '_')}={verified_rank}, "
            f"tolerance={tolerance:.6g}."
        )


@dataclass(frozen=True)
class PiPLSCoreResult:
    """Result of the fixed-parameter Π-PLS construction.

    Attributes
    ----------
    Pi:
        Retained predictor basis with shape ``(p, r_pi)``.
    C:
        Response basis with shape ``(q, h)``.
    W:
        Least-squares map with shape ``(r_pi, h)``.
    P:
        Orthonormal predictor directions with shape ``(p, h)``.
    D:
        Nonnegative diagonal dilation matrix with shape ``(h, h)``.
    Q:
        Orthonormal response directions with shape ``(q, h)``.
    x_rank:
        Numerical rank of the supplied predictor matrix for full SVD, or a
        verified lower bound equal to the number of retained nonzero singular
        values for randomized SVD.
    x_rank_is_exact:
        Whether ``x_rank`` is the complete numerical rank rather than a lower
        bound from a truncated randomized decomposition.
    rank_tolerance:
        Relative-scale tolerance used to assess retained predictor singular
        values.
    predictor_svd_solver:
        Predictor decomposition actually used: ``"full"`` or ``"randomized"``.
    """

    Pi: FloatArray
    C: FloatArray
    W: FloatArray
    P: FloatArray
    D: FloatArray
    Q: FloatArray
    x_rank: int
    x_rank_is_exact: bool
    rank_tolerance: float
    predictor_svd_solver: ResolvedSVDSolver

    @property
    def standardized_regression_map(self) -> FloatArray:
        """Return the centered/scaled regression map with shape ``(p, q)``."""

        standardized_regression_map: FloatArray = self.P @ self.D @ self.Q.T
        return standardized_regression_map


def _cross_covariance_response_basis(
    Z: FloatArray,
    Y: FloatArray,
    *,
    n_components: int,
) -> FloatArray:
    """Return the published cross-covariance-driven response basis."""

    cross_product = Z.T @ Y
    _, _, cross_vt = np.linalg.svd(cross_product, full_matrices=False)
    return np.asarray(cross_vt[:n_components, :].T, dtype=np.float64)


def _least_squares_response_basis(
    Z: FloatArray,
    Y: FloatArray,
    *,
    n_components: int,
) -> FloatArray:
    """Return the least-squares/RRR-inspired response basis."""

    # Software extension: this least-squares/RRR-inspired response-subspace
    # construction is not part of the peer-reviewed companion publication.
    Z_basis, _ = np.linalg.qr(Z, mode="reduced")
    projected_response = Z_basis.T @ Y
    _, _, response_vt = np.linalg.svd(projected_response, full_matrices=False)
    return np.asarray(response_vt[:n_components, :].T, dtype=np.float64)


def fit_pipls_core(
    X: ArrayLike,
    Y: ArrayLike,
    *,
    predictor_rank: int,
    n_components: int,
    response_subspace: ResponseSubspace = "cross_covariance",
    svd_solver: SVDSolver = "full",
    random_state: int | np.random.RandomState | None = 0,
) -> PiPLSCoreResult:
    r"""Fit the fixed-parameter Π-PLS core to preprocessed matrices.

    Parameters
    ----------
    X:
        Centered, optionally scaled predictor matrix with shape ``(n, p)``.
    Y:
        Centered, optionally scaled response matrix with shape ``(n, q)``.
    predictor_rank:
        Retained predictor-subspace dimension $r_\pi$.
    n_components:
        Number of paired latent modes $h$.
    response_subspace:
        Private response-basis construction policy. ``"cross_covariance"``
        uses the peer-reviewed construction; ``"least_squares"`` uses the
        least-squares/RRR-inspired software extension from Decision 0155.
    svd_solver:
        Predictor decomposition policy. ``"full"`` uses NumPy's exact thin SVD,
        ``"randomized"`` uses scikit-learn's randomized truncated SVD, and
        ``"auto"`` applies the package's conservative size/rank rule.
    random_state:
        Integer seed, NumPy ``RandomState`` instance, or ``None``. An integer
        gives reproducible randomized SVD; ``None`` uses NumPy's global random
        state, following scikit-learn convention.

    Returns
    -------
    PiPLSCoreResult
        The complete private factorization used by later estimator layers.

    Raises
    ------
    ValueError
        If arrays are nonfinite or incompatible, dimensions are inadmissible,
        the requested predictor rank exceeds the verified numerical rank, or
        the SVD policy is invalid.
    """

    X_array = _as_finite_matrix(X, name="X")
    Y_array = _as_finite_matrix(Y, name="Y")

    if X_array.shape[0] != Y_array.shape[0]:
        raise ValueError(
            "X and Y must contain the same number of samples: "
            f"got {X_array.shape[0]} and {Y_array.shape[0]}."
        )

    n_samples, n_features = X_array.shape
    n_targets = Y_array.shape[1]
    if n_samples == 0 or n_features == 0 or n_targets == 0:
        raise ValueError("X and Y must have at least one sample and one column.")

    r_pi = _as_positive_int(predictor_rank, name="predictor_rank")
    h = _as_positive_int(n_components, name="n_components")
    resolved_response_subspace = _validate_response_subspace(response_subspace)

    if h > min(r_pi, n_targets):
        raise ValueError(
            "n_components must satisfy n_components <= min(predictor_rank, n_targets); "
            f"got n_components={h}, predictor_rank={r_pi}, n_targets={n_targets}."
        )

    (
        Pi,
        x_rank,
        x_rank_is_exact,
        rank_tolerance,
        predictor_svd_solver,
    ) = _decompose_predictors(
        X_array,
        predictor_rank=r_pi,
        svd_solver=svd_solver,
        random_state=random_state,
    )
    if r_pi > x_rank:
        raise _PredictorRankInfeasibleError(
            requested_rank=r_pi,
            verified_rank=x_rank,
            rank_is_exact=x_rank_is_exact,
            tolerance=rank_tolerance,
        )

    Z = X_array @ Pi

    if resolved_response_subspace == "cross_covariance":
        C = _cross_covariance_response_basis(
            Z,
            Y_array,
            n_components=h,
        )
    else:
        C = _least_squares_response_basis(
            Z,
            Y_array,
            n_components=h,
        )

    Y_C = Y_array @ C
    W_raw, _, _, _ = np.linalg.lstsq(Z, Y_C, rcond=None)
    W = cast(FloatArray, np.asarray(W_raw, dtype=np.float64))

    M, dilation, N_t = np.linalg.svd(W, full_matrices=False)
    M = np.asarray(M[:, :h], dtype=np.float64)
    dilation = np.asarray(dilation[:h], dtype=np.float64)
    N = np.asarray(N_t[:h, :].T, dtype=np.float64)

    P = np.asarray(Pi @ M, dtype=np.float64)
    D = np.diag(dilation).astype(np.float64, copy=False)
    Q = np.asarray(C @ N, dtype=np.float64)

    return PiPLSCoreResult(
        Pi=Pi,
        C=C,
        W=W,
        P=P,
        D=D,
        Q=Q,
        x_rank=x_rank,
        x_rank_is_exact=x_rank_is_exact,
        rank_tolerance=rank_tolerance,
        predictor_svd_solver=predictor_svd_solver,
    )


def _decompose_predictors(
    X: ArrayLike,
    *,
    predictor_rank: int,
    svd_solver: SVDSolver,
    random_state: int | np.random.RandomState | None,
) -> tuple[FloatArray, int, bool, float, ResolvedSVDSolver]:
    """Return the retained predictor basis and verified rank evidence."""

    X_array = _as_finite_matrix(X, name="X")
    n_samples, n_features = X_array.shape
    r_pi = _as_positive_int(predictor_rank, name="predictor_rank")
    algebraic_limit = min(n_samples, n_features)
    if r_pi > algebraic_limit:
        raise ValueError(
            "predictor_rank must satisfy predictor_rank <= min(n_samples, n_features); "
            f"got predictor_rank={r_pi}, min(...)={algebraic_limit}."
        )

    resolved_solver = _resolve_predictor_svd_solver(
        shape=(n_samples, n_features),
        predictor_rank=r_pi,
        svd_solver=svd_solver,
    )
    validated_random_state = _validate_random_state(random_state)
    if resolved_solver == "full":
        _, x_singular_values, x_vt = np.linalg.svd(X_array, full_matrices=False)
        x_rank_is_exact = True
    else:
        _, x_singular_values, x_vt = randomized_svd(
            X_array,
            n_components=r_pi,
            n_iter="auto",
            random_state=validated_random_state,
            flip_sign=True,
        )
        x_singular_values = np.asarray(x_singular_values, dtype=np.float64)
        x_vt = np.asarray(x_vt, dtype=np.float64)
        x_rank_is_exact = False

    rank_tolerance = _svd_rank_tolerance(
        (n_samples, n_features),
        x_singular_values,
    )
    x_rank = int(np.count_nonzero(x_singular_values > rank_tolerance))
    return (
        np.asarray(x_vt[:r_pi, :].T, dtype=np.float64),
        x_rank,
        x_rank_is_exact,
        rank_tolerance,
        resolved_solver,
    )


def _resolve_predictor_svd_solver(
    *,
    shape: tuple[int, int],
    predictor_rank: int,
    svd_solver: SVDSolver,
) -> ResolvedSVDSolver:
    """Resolve the predictor SVD policy using the conservative auto rule."""

    if svd_solver not in ("full", "randomized", "auto"):
        raise ValueError(f'svd_solver must be "full", "randomized", or "auto"; got {svd_solver!r}.')
    if svd_solver != "auto":
        return svd_solver

    n_samples, n_features = shape
    min_dimension = min(n_samples, n_features)
    use_randomized = (
        min_dimension >= _AUTO_RANDOMIZED_MIN_DIMENSION
        and n_samples * n_features >= _AUTO_RANDOMIZED_MIN_ENTRIES
        and predictor_rank <= _AUTO_RANDOMIZED_MAX_RANK_FRACTION * min_dimension
    )
    return "randomized" if use_randomized else "full"


def _validate_response_subspace(response_subspace: object) -> ResponseSubspace:
    """Validate and narrow the private response-subspace policy."""

    if not isinstance(response_subspace, str) or response_subspace not in (
        "cross_covariance",
        "least_squares",
    ):
        raise ValueError(
            'response_subspace must be "cross_covariance" or "least_squares"; '
            f"got {response_subspace!r}."
        )
    return cast(ResponseSubspace, response_subspace)


def _validate_random_state(
    random_state: object,
) -> int | np.random.RandomState | None:
    """Validate the scikit-learn-style random-state parameter."""

    if random_state is None or isinstance(random_state, np.random.RandomState):
        return random_state
    if isinstance(random_state, (bool, np.bool_)) or not isinstance(
        random_state,
        (int, np.integer),
    ):
        raise ValueError(
            "random_state must be None, an integer in "
            f"[0, {_MAX_RANDOM_STATE}], or a numpy.random.RandomState instance; "
            f"got {random_state!r}."
        )
    seed = int(random_state)
    if seed < 0 or seed > _MAX_RANDOM_STATE:
        raise ValueError(
            "random_state must be None, an integer in "
            f"[0, {_MAX_RANDOM_STATE}], or a numpy.random.RandomState instance; "
            f"got {random_state!r}."
        )
    return seed


def _as_finite_matrix(value: ArrayLike, *, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array; got ndim={array.ndim}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _as_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer; got {value!r}.")
    integer = int(value)
    if integer < 1:
        raise ValueError(f"{name} must be a positive integer; got {value!r}.")
    return integer


def _svd_rank_tolerance(shape: tuple[int, int], singular_values: FloatArray) -> float:
    if singular_values.size == 0:
        return 0.0
    return float(max(shape) * np.finfo(np.float64).eps * singular_values[0])
