from dataclasses import dataclass, replace
from typing import Optional

import jax
import jax.numpy as jnp
from jax import lax

from ..types import BoxModel
from ..utils import convert_scalar_trajectory, extract_segment, register_dataclass_pytree

@register_dataclass_pytree
@dataclass(frozen=True)
class SaddleNodeConfig:
    """Configuration for the stochastic saddle-node system."""

    epsilon: float = 1.25
    lambda_max: float = 3.0
    dt: float = 0.01
    t_start: float = -16.0
    t_end: float = 8.0
    initial_state_min: float = -10.0
    initial_state_max: float = -1.0
    tip_threshold: float = 0.0

    @classmethod
    def default(cls) -> "SaddleNodeConfig":
        """Return default configuration."""
        return cls()




class SaddleNodeModel(BoxModel):
    """BoxModel abstraction for the stochastic saddle-node system."""

    def __init__(self, config: SaddleNodeConfig):
        self._config = config

    def get_parameters(self):
        """Get the parameters of the model."""
        return {
            "epsilon": self._config.epsilon,
            "lambda_max": self._config.lambda_max,
            "dt": self._config.dt,
            "t_start": self._config.t_start,
            "t_end": self._config.t_end,
            "initial_state_min": self._config.initial_state_min,
            "initial_state_max": self._config.initial_state_max,
            "tip_threshold": self._config.tip_threshold,
        }

    def _lambda_schedule(
        self,
        t: jnp.ndarray,
        config: SaddleNodeConfig,
    ):
        """Compute the lambda schedule."""
        return 0.5 * config.lambda_max * (
            jnp.tanh(0.5 * config.lambda_max * config.epsilon * t) + 1.0
        )

    def _sample_initial_state(
        self,
        key: jax.random.PRNGKey,
        config: SaddleNodeConfig,
    ):
        """Sample an initial state from the uniform distribution."""
        return float(
            jax.random.uniform(
                key,
                (),
                minval=config.initial_state_min,
                maxval=config.initial_state_max,
            )
        )

    def _simulate_path(
        self,
        key: jax.random.PRNGKey,
        lambda_t: jnp.ndarray,
        dt: float,
        noise_scale: float,
        initial_state: float,
        noise: Optional[jnp.ndarray] = None,
    ):
        """Stochastic integration."""

        if noise is None:
            noise = jax.random.normal(key, shape=(lambda_t.shape[0] - 1,))
        else:
            noise = jnp.asarray(noise)

        def step(x_prev, inputs):
            lam_t, noise_t = inputs
            det_part = ((x_prev + lam_t) ** 2 - 1.0) * dt
            x_next = x_prev + det_part + noise_scale * noise_t
            return x_next, x_next

        _, x_tail = lax.scan(step, initial_state, (lambda_t[:-1], noise))
        x0 = jnp.full((1,), initial_state, dtype=lambda_t.dtype)
        return jnp.concatenate([x0, x_tail], axis=0)


    def simulate(
        self,
        key: jax.random.PRNGKey,
        total_len: int,
        sigma_tilde: float = 0.01,
        initial_state: Optional[jnp.ndarray] = None,
        noise: Optional[jnp.ndarray] = None,
        null_simulation: bool = False,
    ):
        """Simulate the saddle-node system.

        For null_simulation=True, same ramp formula but epsilon at 50% (slower rate).
        """
        key_noise, key_ic = jax.random.split(key)

        # Setup time and lambda schedule
        self._config = replace(self._config, t_start=-(total_len * self._config.dt))
        time = jnp.arange(self._config.t_start, self._config.t_end, self._config.dt)
        if null_simulation:
            config_null = replace(self._config, epsilon=0.5 * self._config.epsilon)
            lambda_t = self._lambda_schedule(time, config_null)
        else:
            lambda_t = self._lambda_schedule(time, self._config)

        # Randomly sample initial state if not provided
        if initial_state is None:
            initial_state = self._sample_initial_state(key_ic, self._config)
        else:
            initial_state = jnp.asarray(initial_state)

        # Configure noise scale
        noise_scale = jnp.sqrt(2.0 * sigma_tilde * self._config.dt)

        # Simulate trajectory
        trajectory = self._simulate_path(
            key_noise,
            lambda_t,
            self._config.dt,
            noise_scale,
            initial_state,
            noise=noise,
        )

        # Post-process trajectory
        trajectory = convert_scalar_trajectory(trajectory)

        return time, trajectory, lambda_t, noise_scale

    def is_tipped(
        self,
        trajectory: jnp.ndarray,
        **kwargs,
    ):
        """Check if the trajectory is tipped."""
        scalar_view = trajectory[:, 0] if trajectory.ndim > 1 else trajectory
        return bool(jnp.any(scalar_view > self._config.tip_threshold))

    def locate_transition(
        self,
        trajectory: jnp.ndarray,
        null_simulation: bool,
        **kwargs,
    ):
        """Locate the transition index."""
        scalar_view = trajectory[:, 0] if trajectory.ndim > 1 else trajectory
        if null_simulation:
            return int(scalar_view.shape[0] // 2)

        crossed = jnp.where(scalar_view > self._config.tip_threshold, 1, 0)
        has_crossed = bool(jnp.any(crossed))
        if not has_crossed:
            return None
        return int(jnp.argmax(crossed))

    def extract_segment(
        self,
        time: jnp.ndarray,
        trajectory: jnp.ndarray,
        param_values: jnp.ndarray,
        transition_idx: int,
        steps_before_transition: int,
        series_len: int,
    ):
        """Extract a segment of the trajectory around the transition."""
        time_seg, traj_seg, param_seg, trans_idx_out, success = extract_segment(
            time,
            trajectory,
            param_values,
            transition_idx,
            steps_before_transition,
            series_len,
        )

        return time_seg, traj_seg, param_seg, trans_idx_out, success
