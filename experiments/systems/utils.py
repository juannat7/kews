from dataclasses import fields
from typing import Tuple, Type, TypeVar

import jax
import jax.numpy as jnp

T = TypeVar("T")


def register_dataclass_pytree(cls: Type[T]) -> Type[T]:
    """Register model parameters as a JAX pytree in dataclass field order."""

    def flatten(instance: T):
        children = tuple(getattr(instance, f.name) for f in fields(instance))
        return children, None

    def unflatten(aux_data, children):
        return cls(*children)

    jax.tree_util.register_pytree_node(cls, flatten, unflatten)
    return cls


def convert_scalar_trajectory(x: jnp.ndarray) -> jnp.ndarray:
    """Lift a scalar trajectory to a 2D array [x, 0] to reuse 2D utilities."""
    return jnp.stack([x, jnp.zeros_like(x)], axis=-1)


def calculate_total_len(
    series_len: int,
    buffer_size: int,
    param_end_multiplier: float = 1.1
) -> int:
    """Calculate total simulation length with optional extension.
    
    For simulations with param_end_multiplier > 1.0, extends the simulation time
    to ensure enough points before transition (forced) or after transition (null).
    
    Parameters
    ----------
    series_len : int
        Desired output trajectory length
    buffer_size : int
        Extra samples simulated prior to the transition
    param_end_multiplier : float, default=1.1
        Multiplier for extending simulation time beyond transition
        
    Returns
    -------
    int
        Total simulation length (extended if param_end_multiplier > 1.0)
    """
    base_len = series_len + buffer_size
    if param_end_multiplier > 1.0:
        return int(base_len * param_end_multiplier)
    return base_len


def extract_segment(
    time: jnp.ndarray,
    trajectory: jnp.ndarray,
    params: jnp.ndarray,
    transition_idx: int,
    steps_before_transition: int,
    series_len: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int, bool]:
    """Extract a segment mirroring bifurcation.extract_pre_transition_series."""

    gap = max(steps_before_transition, 0)
    end_idx = transition_idx - gap
    start_idx = end_idx - series_len
    success = (end_idx > 0) and (end_idx >= series_len) and (start_idx >= 0)

    if success:
        time_out = time[start_idx:end_idx] - time[start_idx]
        trajectory_out = trajectory[start_idx:end_idx]
        params_out = params[start_idx:end_idx]
    else:
        time_out = time[:series_len] - time[0]
        trajectory_out = trajectory[:series_len]
        params_out = params[:series_len]

    return time_out, trajectory_out, params_out, transition_idx, success

