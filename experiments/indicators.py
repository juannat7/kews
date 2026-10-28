"""Classical and spectral comparison indicators for the paper notebooks."""

import jax
import jax.numpy as jnp
import numpy as np

from kews.diagnostics import fit_operator, generator_eigenvalues
from kews.observables import time_delay


def window_size(n, window):
    """Resolve a fractional or absolute rolling window length."""
    size = int(window * n) if window <= 1 else int(window)
    if not 0 < size < n:
        raise ValueError(f"Window {size} must be in (0, {n})")
    return size


def _rolling(data, window, statistic):
    data = jnp.asarray(data)
    if data.ndim != 1:
        raise ValueError("Classical indicators require a one-dimensional series")
    size = window_size(len(data), window)
    indices = jnp.arange(len(data) - size + 1)[:, None] + jnp.arange(size)
    values = jax.vmap(statistic)(data[indices])
    return jnp.full(data.shape, jnp.nan, dtype=jnp.float32).at[size - 1:].set(values)


def rolling_variance(data, window=0.5):
    """Population variance, aligned to the right edge of each window."""
    return _rolling(data, window, jax.jit(lambda x: x.var()))


def _correlation(data, window, lag, pearson):
    if not isinstance(lag, int) or not 0 < lag < window_size(len(data), window):
        raise ValueError("lag must be a positive integer smaller than the window")

    @jax.jit
    def statistic(values):
        x, y = values[:-lag], values[lag:]
        x, y = x - x.mean(), y - y.mean()
        denominator = jnp.sqrt((x ** 2).sum() * (y ** 2).sum()) if pearson else (x ** 2).sum()
        return jnp.where(denominator > 0, (x * y).sum() / denominator, 0.0)

    return _rolling(data, window, statistic)


def rolling_ar1(data, window=0.5, lag=1):
    """Lagged regression slope used in the coupled AMOC analysis."""
    return _correlation(data, window, lag, pearson=False)


def rolling_acf(data, window=0.5, lag=1):
    """Lagged Pearson correlation used in the prototypical model analysis."""
    return _correlation(data, window, lag, pearson=True)


def leading_decay_rate(eigvals_L, tol=1e-10):
    """Return the stable non-constant generator rate closest to zero."""
    lam_L = np.asarray(eigvals_L)
    sigma = np.real(lam_L)
    omega = np.imag(lam_L)
    # Exclude the constant mode and growing modes from the stable decay rate.
    valid = (
        np.isfinite(sigma)
        & np.isfinite(omega)
        & (np.abs(lam_L) > tol)
        & (sigma < -tol)
    )
    if not np.any(valid):
        return np.nan
    return np.max(sigma[valid])


def spectral_gap_series(traj, wL, dim_delay=5, low_rank=0.9, dt=1.0, tol=1e-10):
    """Return the rolling EDMD gap, ``-Re(lambda_1)``, in inverse time units.

    Each window is centered and standardized before fitting delay-coordinate
    EDMD. ``dt`` is the sampling interval; ``low_rank`` controls SVD truncation.
    Results are aligned to window endpoints. Constant or nonfinite windows,
    and windows without a stable non-constant mode, retain ``NaN``.
    """
    data = np.asarray(traj, dtype=float)
    n = len(data)
    gap = np.full(n, np.nan, dtype=float)
    w = window_size(n, wL)
    for start in range(n - w + 1):
        end_idx = start + w - 1
        window_vals = data[start : start + w]
        if not np.all(np.isfinite(window_vals)):
            continue
        window_vals = window_vals - np.mean(window_vals)
        window_std = np.std(window_vals)
        if not np.isfinite(window_std) or window_std <= tol:
            continue
        window_vals = window_vals / window_std
        out = fit_operator(
            jnp.array(window_vals), observable=time_delay(dim_delay), rank=low_rank,
        )
        lam_L = np.asarray(generator_eigenvalues(out["eigvals"], dt=dt))
        sigma_1 = leading_decay_rate(lam_L, tol=tol)
        if np.isfinite(sigma_1):
            gap[end_idx] = -sigma_1
    return gap
