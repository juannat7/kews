"""Koopman and ResDMD numerical kernels.

Snapshot matrices have shape ``(features, time)``; rolling evaluation adds a
leading window axis. Normalization and numerical tolerances follow the paper
reference so that refactoring does not change the plotted indicators.
"""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any, Literal, Mapping, Optional, Union

import jax
import jax.numpy as jnp

Rank = Union[int, float]


def delay_embed(x: jax.Array, n_delay: int) -> jax.Array:
    """Stack delayed state copies with time on the final axis."""
    if not isinstance(n_delay, Integral) or isinstance(n_delay, bool) or n_delay < 1:
        raise ValueError("n_delay must be a positive integer")
    x = jnp.atleast_2d(x)
    n_time = x.shape[1]
    if n_delay > n_time:
        raise ValueError(
            f"n_delay must not exceed the time length ({n_time}), got {n_delay}"
        )
    return jnp.vstack(
        [x[:, offset : n_time - n_delay + offset + 1] for offset in range(n_delay)]
    )


def random_fourier_features(
    x: jax.Array,
    n_features: int,
    scale: float,
    key: jax.Array,
) -> jax.Array:
    """Map state columns to random cosine features."""
    x = jnp.atleast_2d(x)
    projection_key, phase_key = jax.random.split(key)
    weights = jax.random.normal(projection_key, (n_features, x.shape[0])) / scale
    phase = jax.random.uniform(
        phase_key, (n_features,), maxval=2 * jnp.pi
    )
    return jnp.sqrt(2 / n_features) * jnp.cos(weights @ x + phase[:, None])


def _rank(singular_values: jax.Array, low_rank: Rank) -> int:
    """Resolve an energy fraction or absolute SVD rank to a concrete integer."""
    n_modes = singular_values.shape[0]
    if isinstance(low_rank, Integral) and not isinstance(low_rank, bool):
        if low_rank < 1:
            raise ValueError("rank must be a fraction in (0, 1) or an integer >= 1")
        return min(int(low_rank), n_modes)
    if isinstance(low_rank, Real) and 0 < low_rank < 1:
        energy = singular_values**2
        cumulative = jnp.cumsum(energy) / jnp.maximum(jnp.sum(energy), 1e-12)
        index = int(jax.device_get(jnp.searchsorted(cumulative, low_rank)))
        return min(index + 1, n_modes)
    raise ValueError("rank must be a fraction in (0, 1) or an integer >= 1")


def _svd_inverse(singular_values: jax.Array) -> jax.Array:
    """Return a stable diagonal inverse for retained singular values."""
    return jnp.diag(
        jnp.where(singular_values > 1e-12, 1.0 / singular_values, 0.0)
    )


def _edmd_matrices(
    current: jax.Array,
    future: jax.Array,
    rank: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Build EDMD bases, Gram matrix, and cross matrix."""
    # Preserve the paper's snapshot normalization before truncating the SVD.
    _, singular_values, vh = jnp.linalg.svd(
        current / current.shape[1], full_matrices=False
    )
    values = singular_values[:rank]
    right = vh[:rank].T
    inverse = _svd_inverse(values)
    basis_current = current @ right @ inverse
    basis_future = future @ right @ inverse
    gram = basis_current.T @ basis_current
    gram = (gram + gram.T) / 2
    cross = basis_current.T @ basis_future
    return basis_current, basis_future, gram, cross


def _mean_residual(
    gram: jax.Array,
    cross: jax.Array,
    future_gram: jax.Array,
    operator: jax.Array,
) -> jax.Array:
    """Average the ResDMD residual over Koopman eigenpairs."""
    eigenvalues, eigenvectors = jnp.linalg.eig(operator)

    def one(eigenvalue: jax.Array, eigenvector: jax.Array) -> jax.Array:
        # Residual energy is normalized by this eigenfunction's Gram norm.
        residual_matrix = (
            future_gram
            - eigenvalue * cross.conj().T
            - eigenvalue.conj() * cross
            + jnp.abs(eigenvalue) ** 2 * gram
        )
        conjugate = eigenvector.conj()
        numerator = (conjugate @ residual_matrix @ eigenvector).real
        denominator = (conjugate @ gram @ eigenvector).real
        return jnp.where(
            jnp.abs(denominator) > 1e-10,
            jnp.abs(numerator / denominator),
            0.0,
        )

    return jnp.mean(jax.vmap(one)(eigenvalues, eigenvectors.T))


def _edmd_residual(current: jax.Array, future: jax.Array, rank: int) -> jax.Array:
    """Compute one autonomous or augmented EDMD residual."""
    _, basis_future, gram, cross = _edmd_matrices(current, future, rank)
    future_gram = basis_future.T @ basis_future
    future_gram = (future_gram + future_gram.T) / 2
    operator = jnp.linalg.pinv(gram, rtol=1e-4) @ cross
    residual = _mean_residual(gram, cross, future_gram, operator)
    valid = jnp.all(jnp.isfinite(operator)) & jnp.isfinite(residual)
    return jnp.where(valid, residual, 0.0)


def _dmdc(
    current: jax.Array,
    future: jax.Array,
    control: jax.Array,
    input_rank: int,
    output_rank: int,
) -> tuple[jax.Array, jax.Array]:
    """Fit DMD with control from aligned snapshot matrices."""
    n_state = current.shape[0]
    omega = jnp.vstack([current, control])
    left, singular_values, vh = jnp.linalg.svd(omega, full_matrices=False)
    left = left[:, :input_rank]
    projection = future @ vh[:input_rank].T @ _svd_inverse(
        singular_values[:input_rank]
    )
    state_operator = projection @ left[:n_state].T
    control_operator = projection @ left[n_state:].T

    # Match the reduced-order lift used by the paper reference implementation.
    output_basis = jnp.linalg.svd(future, full_matrices=False)[0][:, :output_rank]
    state_operator = (
        output_basis
        @ (output_basis.T @ state_operator @ output_basis)
        @ output_basis.T
    )
    control_operator = output_basis @ (output_basis.T @ control_operator)
    return state_operator, control_operator


def _control_residual(
    current: jax.Array,
    future: jax.Array,
    control: jax.Array,
    input_rank: int,
    output_rank: int,
) -> jax.Array:
    """Compute one DMDc residual in lifted state space."""
    operator, _ = _dmdc(current, future, control, input_rank, output_rank)
    gram = current @ current.T
    cross = current @ future.T
    future_gram = future @ future.T
    future_gram = (future_gram + future_gram.T) / 2
    residual = _mean_residual(gram, cross, future_gram, operator)
    valid = jnp.all(jnp.isfinite(operator)) & jnp.isfinite(residual)
    return jnp.where(valid, residual, 0.0)


def _augment(
    lifted: jax.Array,
    control: jax.Array,
    future_input: Literal["future", "zero"],
) -> tuple[jax.Array, jax.Array]:
    """Build ``[phi(x), phi(x)u, u]`` current and future snapshots."""
    state_current, state_future = lifted[:, :-1], lifted[:, 1:]
    control_current = control[:, :-1]
    if future_input == "future":
        control_future = control[:, 1:]
    elif future_input == "zero":
        control_future = jnp.zeros_like(control_current)
    else:
        raise ValueError("future_input must be 'future' or 'zero'")

    interaction_current = jnp.einsum(
        "it,jt->ijt", state_current, control_current
    ).reshape((-1, state_current.shape[1]))
    interaction_future = jnp.einsum(
        "it,jt->ijt", state_future, control_future
    ).reshape((-1, state_future.shape[1]))
    return (
        jnp.vstack([state_current, interaction_current, control_current]),
        jnp.vstack([state_future, interaction_future, control_future]),
    )


def _lift(observable: Any, x: jax.Array) -> jax.Array:
    """Apply an observable, or pass through features lifted before windowing."""
    return x if observable is None else observable.transform(x)


def rolling_residuals(
    windows: jax.Array,
    observable: Any,
    low_rank: Rank,
    control_windows: Optional[jax.Array] = None,
    control_method: Literal["dmdc", "augmented"] = "dmdc",
    future_input: Literal["future", "zero"] = "future",
    fit_kwargs: Optional[Mapping[str, Any]] = None,
) -> jax.Array:
    """Compute one residual for every pre-built rolling window."""
    if getattr(observable, "fit_on_first_window", False):
        # Freeze learned features after the initial window to avoid future-data leakage.
        observable.fit(windows[0], **dict(fit_kwargs or {}))

    first_lifted = _lift(observable, windows[0])
    if first_lifted.shape[1] < 2:
        raise ValueError("observable must produce at least two time columns per window")

    if control_windows is None:
        first_current = first_lifted[:, :-1]
        singular_values = jnp.linalg.svd(
            first_current / first_current.shape[1], full_matrices=False
        )[1]
        rank = _rank(singular_values, low_rank)

        # Hold rank fixed so later windows use the same approximation dimension.
        @jax.jit
        def one(window: jax.Array) -> jax.Array:
            lifted = _lift(observable, window)
            return _edmd_residual(lifted[:, :-1], lifted[:, 1:], rank)

        return jax.vmap(one)(windows)

    if control_method not in ("dmdc", "augmented"):
        raise ValueError("control_method must be 'dmdc' or 'augmented'")

    first_control = control_windows[0]
    first_control = (
        first_control.reshape((1, -1))
        if first_control.ndim == 1
        else first_control
    )
    n_lifted = first_lifted.shape[1]

    # Delay coordinates end at the latest sample; align control to those endpoints.
    if control_method == "dmdc":
        first_current, first_future = first_lifted[:, :-1], first_lifted[:, 1:]
        aligned = first_control[:, first_control.shape[1] - n_lifted : -1]
        input_rank = _rank(
            jnp.linalg.svd(jnp.vstack([first_current, aligned]), full_matrices=False)[1],
            low_rank,
        )
        output_rank = _rank(
            jnp.linalg.svd(first_future, full_matrices=False)[1], low_rank
        )

        @jax.jit
        def one(window: jax.Array, control_window: jax.Array) -> jax.Array:
            lifted = _lift(observable, window)
            control = (
                control_window.reshape((1, -1))
                if control_window.ndim == 1
                else control_window
            )
            aligned = control[:, control.shape[1] - lifted.shape[1] : -1]
            return _control_residual(
                lifted[:, :-1],
                lifted[:, 1:],
                aligned,
                input_rank,
                output_rank,
            )

        return jax.vmap(one)(windows, control_windows)

    first_aligned = first_control[:, first_control.shape[1] - n_lifted :]
    first_current, _ = _augment(first_lifted, first_aligned, future_input)
    rank = _rank(
        jnp.linalg.svd(
            first_current / first_current.shape[1], full_matrices=False
        )[1],
        low_rank,
    )

    @jax.jit
    def one(window: jax.Array, control_window: jax.Array) -> jax.Array:
        lifted = _lift(observable, window)
        control = (
            control_window.reshape((1, -1))
            if control_window.ndim == 1
            else control_window
        )
        aligned = control[:, control.shape[1] - lifted.shape[1] :]
        current, future = _augment(lifted, aligned, future_input)
        return _edmd_residual(current, future, rank)

    return jax.vmap(one)(windows, control_windows)


def fit_koopman_edmd(
    x: jax.Array,
    observable: Any,
    low_rank: Rank = 0.9,
    fit_kwargs: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Fit an autonomous Koopman operator in a supplied observable space."""
    x = jnp.asarray(x)
    if getattr(observable, "fit_on_first_window", False):
        observable.fit(x, **dict(fit_kwargs or {}))
    lifted = _lift(observable, x)
    if lifted.shape[1] < 2:
        raise ValueError("observable must produce at least two time columns")
    current, future = lifted[:, :-1], lifted[:, 1:]
    singular_values = jnp.linalg.svd(
        current / current.shape[1], full_matrices=False
    )[1]
    rank = _rank(singular_values, low_rank)
    basis_current, basis_future, gram, cross = _edmd_matrices(
        current, future, rank
    )
    operator = jnp.linalg.lstsq(gram, cross, rcond=1e-10)[0]
    eigenvalues, eigenvectors = jnp.linalg.eig(operator)
    return {
        "K": operator,
        "eigvals": eigenvalues,
        "eigvecs": eigenvectors.T,
        "Psi_X": basis_current.T @ current,
        "Psi_Y": basis_current.T @ future,
        "r": rank,
    }


def _aligned_control(x: jax.Array, u: jax.Array) -> jax.Array:
    """Normalize control orientation and validate raw time alignment."""
    u = jnp.atleast_2d(u)
    if u.shape[1] != x.shape[-1]:
        raise ValueError("control must have the same time length as state")
    return u


def compute_dmdc(
    x: jax.Array,
    u: jax.Array,
    dim_delay: int = 5,
    low_rank: Rank = 0.9,
) -> tuple[jax.Array, jax.Array]:
    """Fit DMDc in delay-coordinate state space."""
    x = jnp.atleast_1d(x).flatten()
    u = _aligned_control(x, u)
    if x.size < dim_delay + 2:
        raise ValueError(f"data needs at least {dim_delay + 2} time samples")
    lifted = delay_embed(x, dim_delay)
    current, future = lifted[:, :-1], lifted[:, 1:]
    control = u[:, dim_delay - 1 : -1]
    input_rank = _rank(
        jnp.linalg.svd(jnp.vstack([current, control]), full_matrices=False)[1],
        low_rank,
    )
    output_rank = _rank(jnp.linalg.svd(future, full_matrices=False)[1], low_rank)
    return _dmdc(current, future, control, input_rank, output_rank)


def compute_edmdwc(
    x: jax.Array,
    u: jax.Array,
    dim_delay: int = 5,
    low_rank: Rank = 0.9,
) -> tuple[jax.Array, jax.Array]:
    """Fit reduced EDMD with control in delay-coordinate state space."""
    x = jnp.atleast_1d(x).flatten()
    u = _aligned_control(x, u)
    if x.size < dim_delay + 2:
        raise ValueError(f"data needs at least {dim_delay + 2} time samples")
    lifted = delay_embed(x, dim_delay)
    current, future = lifted[:, :-1], lifted[:, 1:]
    control = u[:, dim_delay - 1 : -1]
    left, singular_values, _ = jnp.linalg.svd(current, full_matrices=False)
    rank = _rank(singular_values, low_rank)
    basis = left[:, :rank]
    current_reduced, future_reduced = basis.T @ current, basis.T @ future
    fit = future_reduced @ jnp.linalg.pinv(
        jnp.vstack([current_reduced, control]), rtol=1e-4
    )
    return (
        basis @ fit[:, :rank] @ basis.T,
        basis @ fit[:, rank:],
    )


def generator_eigenvalues(eigenvalues: jax.Array, dt: float = 1.0) -> jax.Array:
    """Convert discrete Koopman eigenvalues to infinitesimal-generator values."""
    if not isinstance(dt, Real) or dt <= 0:
        raise ValueError("dt must be positive")
    eigenvalues = jnp.asarray(eigenvalues)
    safe = jnp.where(jnp.abs(eigenvalues) < 1e-12, jnp.nan + 0j, eigenvalues)
    return jnp.log(safe) / dt


def generator_dominant_mode(
    eigenvalues: jax.Array, dt: float = 1.0
) -> dict[str, Any]:
    """Summarize the slowest non-constant finite generator mode."""
    if not isinstance(dt, Real) or dt <= 0:
        raise ValueError("dt must be positive")
    eigenvalues = jnp.asarray(eigenvalues)
    finite = jnp.isfinite(eigenvalues.real) & jnp.isfinite(eigenvalues.imag)
    valid = finite & (jnp.abs(eigenvalues) > 1e-10)
    if not bool(jnp.any(valid)):
        return {
            "decay_rate": float("nan"),
            "angular_freq": float("nan"),
            "period": float("nan"),
            "e_folding_time": float("nan"),
            "is_oscillatory": False,
            "dominant_idx": -1,
        }
    index = int(jnp.argmin(jnp.where(valid, jnp.abs(eigenvalues.real), jnp.inf)))
    value = eigenvalues[index]
    decay = float(value.real)
    frequency = float(value.imag)
    period = 2 * jnp.pi / abs(frequency) if abs(frequency) > 1e-10 else jnp.nan
    period = float(period)
    return {
        "decay_rate": decay,
        "angular_freq": frequency,
        "period": period,
        "e_folding_time": 1 / abs(decay) if decay < -1e-10 else float("nan"),
        "is_oscillatory": bool(2 * dt <= period <= 1e4 * dt),
        "dominant_idx": index,
    }
