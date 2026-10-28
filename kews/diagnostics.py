"""Optional operator and spectrum diagnostics kept outside the main API."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Union

import jax

from ._core import (
    compute_dmdc as _dmdc,
    compute_edmdwc as _edmdc,
    fit_koopman_edmd as _fit_operator,
    generator_dominant_mode as _dominant_mode,
    generator_eigenvalues as _generator_eigenvalues,
)
from .observables import time_delay


def fit_operator(
    data: jax.Array,
    observable: Optional[Any] = None,
    rank: Union[int, float] = 0.9,
    fit_kwargs: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Fit an autonomous EDMD Koopman operator for diagnostics.

    Args:
        data: State trajectory shaped ``(time,)`` or ``(state, time)``.
        observable: Observable from :mod:`kews.observables`; defaults to
            ``time_delay()``.
        rank: Retained SVD energy fraction or absolute rank.
        fit_kwargs: Optional training arguments for learned observables.

    Returns:
        Dictionary containing ``K``, ``eigvals``, row-wise ``eigvecs``,
        ``Psi_X``, ``Psi_Y``, and the retained rank ``r``.
    """
    return _fit_operator(
        data,
        time_delay() if observable is None else observable,
        rank,
        fit_kwargs,
    )


def dmdc(
    data: jax.Array,
    control: jax.Array,
    n_delay: int = 5,
    rank: Union[int, float] = 0.9,
) -> tuple[jax.Array, jax.Array]:
    """Fit Dynamic Mode Decomposition with control.

    Args:
        data: Scalar state trajectory shaped ``(time,)``.
        control: Input trajectory shaped ``(time,)`` or ``(control, time)``.
        n_delay: Delay-coordinate dimension for state.
        rank: Retained SVD energy fraction or absolute rank.

    Returns:
        Tuple ``(A, B)`` for ``x_next = A @ x + B @ u`` in delay space.
    """
    return _dmdc(data, control, n_delay, rank)


def edmdc(
    data: jax.Array,
    control: jax.Array,
    n_delay: int = 5,
    rank: Union[int, float] = 0.9,
) -> tuple[jax.Array, jax.Array]:
    """Fit reduced EDMD with control in delay-coordinate space.

    Args:
        data: Scalar state trajectory shaped ``(time,)``.
        control: Input trajectory shaped ``(time,)`` or ``(control, time)``.
        n_delay: Delay-coordinate dimension for state.
        rank: Retained SVD energy fraction or absolute rank.

    Returns:
        Tuple ``(K, B)`` containing lifted state and control operators.
    """
    return _edmdc(data, control, n_delay, rank)


def generator_eigenvalues(
    eigenvalues: jax.Array, dt: float = 1.0
) -> jax.Array:
    """Convert discrete Koopman eigenvalues to generator eigenvalues.

    Args:
        eigenvalues: Complex eigenvalues of a discrete Koopman operator.
        dt: Positive sampling interval.

    Returns:
        Complex generator values ``log(eigenvalues) / dt``. Values near zero
        map to ``NaN``.
    """
    return _generator_eigenvalues(eigenvalues, dt)


def dominant_mode(eigenvalues: jax.Array, dt: float = 1.0) -> dict[str, Any]:
    """Summarize the slowest finite non-constant generator mode.

    Args:
        eigenvalues: Complex generator eigenvalues.
        dt: Positive sampling interval used for oscillation bounds.

    Returns:
        Decay rate, angular frequency, period, e-folding time, oscillatory flag,
        and selected eigenvalue index.
    """
    return _dominant_mode(eigenvalues, dt)


__all__ = ["dmdc", "dominant_mode", "edmdc", "fit_operator", "generator_eigenvalues"]
