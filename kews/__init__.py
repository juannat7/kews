"""KEWS: compact Koopman early warning signals."""

from __future__ import annotations

from typing import Any, Literal, Mapping, Optional, Union

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ._core import rolling_residuals
from .observables import _RFF, time_delay


def _window_length(n_time: int, window: Union[int, float]) -> int:
    """Resolve a relative or absolute window length."""
    length = int(window * n_time) if window <= 1 else int(window)
    if not 1 < length < n_time:
        raise ValueError(
            f"window resolves to {length}; expected a value in (1, {n_time})"
        )
    return length


def _windows(data: jax.Array, length: int) -> jax.Array:
    """Return ``(window, time)`` or ``(window, state, time)`` snapshots."""
    indices = jnp.arange(data.shape[-1] - length + 1)[:, None] + jnp.arange(length)
    if data.ndim == 1:
        return data[indices]
    return jnp.transpose(data[:, indices], (1, 0, 2))


def resdmd(
    data: ArrayLike,
    window: Union[int, float] = 0.5,
    observable: Optional[Any] = None,
    rank: Union[int, float] = 0.9,
    control: Optional[ArrayLike] = None,
    *,
    control_method: Literal["dmdc", "augmented"] = "dmdc",
    future_input: Literal["future", "zero"] = "future",
    fit_kwargs: Optional[Mapping[str, Any]] = None,
) -> jax.Array:
    """Compute a rolling Koopman early warning signal.

    The same call covers autonomous and controlled dynamics. Leave ``control``
    unset for autonomous ResDMD or pass an aligned control trajectory to use a
    control-aware formulation. Each residual is aligned to the right edge of
    its window; earlier entries are ``NaN``.

    A learned observable is fitted on the first window and then held fixed.
    The first window also determines the retained rank. Random Fourier
    features use one seeded basis for the complete trajectory.

    Args:
        data: Scalar state shaped ``(time,)`` or multivariate state shaped
            ``(state, time)``.
        window: Fraction of the trajectory in ``(0, 1)`` or absolute sample
            count. It must resolve to at least two samples and less than the
            complete trajectory.
        observable: Observable created by :mod:`kews.observables`. Defaults to
            ``time_delay()``.
        rank: Retained SVD energy fraction in ``(0, 1)`` or an absolute rank.
        control: Optional input shaped ``(time,)`` or ``(control, time)``.
        control_method: ``"dmdc"`` fits DMD with raw control in lifted state
            space. ``"augmented"`` fits ``[phi(x), phi(x)u, u]`` snapshots.
        future_input: For the augmented method, use observed next inputs with
            ``"future"`` or replace them with zero using ``"zero"``.
        fit_kwargs: Training arguments for a learned observable, such as
            ``epochs``, ``batch_size``, and ``lr`` for ``observables.mlp``.

    Returns:
        One-dimensional JAX array with the same time length as ``data``.
        Positions before the first completed window are ``NaN``.

    Raises:
        ValueError: If array orientation, window, rank, control alignment, or a
            method option is invalid.
    """
    data = jnp.asarray(data)
    if data.ndim not in (1, 2):
        raise ValueError("data must have shape (time,) or (state, time)")
    length = _window_length(data.shape[-1], window)

    control_windows = None
    if control is not None:
        control = jnp.asarray(control)
        if control.ndim not in (1, 2) or control.shape[-1] != data.shape[-1]:
            raise ValueError(
                "control must have shape (time,) or (control, time) and match data"
            )
        control_windows = _windows(control, length)

    observable = time_delay() if observable is None else observable
    if not callable(getattr(observable, "transform", None)):
        raise ValueError("observable must come from kews.observables")

    if isinstance(observable, _RFF):
        # Lift once to preserve the paper's shared basis and floating-point evaluation.
        state_windows = _windows(observable.transform(data), length)
        observable = None
    else:
        state_windows = _windows(data, length)

    residuals = rolling_residuals(
        state_windows,
        observable,
        rank,
        control_windows,
        control_method,
        future_input,
        fit_kwargs,
    )
    result = jnp.full((data.shape[-1],), jnp.nan, dtype=jnp.float32)
    return result.at[length - 1 :].set(residuals)


__version__ = "0.1.0"
__all__ = ["resdmd"]
