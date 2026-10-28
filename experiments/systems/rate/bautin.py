from dataclasses import dataclass, replace
from typing import Optional

import jax
import jax.numpy as jnp
from jax import lax

from ..types import BoxModel
from ..utils import extract_segment, register_dataclass_pytree


@register_dataclass_pytree
@dataclass(frozen=True)
class BautinConfig:
    """Configuration for the stochastic Bautin system.

    Based on the equation:
    dZ_t = (a + iω)[Z_t - Λ(t)]dt - b[Z_t - Λ(t)]²[Z_t - Λ(t)]dt
           + [Z_t - Λ(t)]⁴[Z_t - Λ(t)]dt + D₂dW₂t
    where Z_t = X_t + iY_t and Λ(t) = 0.5Λ_max[tanh(0.5Λ_max r t) + 1]
    """

    r: float = 0.1  # Forcing rate (paper default: 0.1)
    lambda_max: float = 8.0  # Maximum forcing amplitude (paper default: 8)
    a: float = 0.1  # Linear coefficient (paper default: 0.1)
    omega: float = 3.0  # Angular frequency (paper default: 3)
    b: float = 1.0  # Cubic coefficient (paper default: 1)
    D2: float = 20  # Noise magnitude (paper default: 0.2, so 20 * 0.01 following default sigma_tilde)
    dt: float = 0.01  # Time step
    t_start: float = -15.0  # Paper (Huang et al. Nat. Mach. Intell. 2024): time range -15 to 15
    t_end: float = 15.0  # Wider window → slower ramp → tipping later; narrower → earlier tipping
    initial_state_min: float = -1.0  # Minimum initial x/y
    initial_state_max: float = 1.0  # Maximum initial x/y
    tip_threshold: float = 10.0  # Threshold for detecting tipping (radius from origin)

    @classmethod
    def default(cls) -> "BautinConfig":
        """Return default configuration."""
        return cls()



class BautinModel(BoxModel):
    """BoxModel abstraction for the stochastic Bautin system.

    Implements the Bautin system:
    dZ_t = (a + iω)[Z_t - Λ(t)]dt - b[Z_t - Λ(t)]²[Z_t - Λ(t)]dt
           + [Z_t - Λ(t)]⁴[Z_t - Λ(t)]dt + D₂dW₂t

    where Z_t = X_t + iY_t (complex state) and
    Λ(t) = 0.5Λ_max[tanh(0.5Λ_max r t) + 1] (time-varying forcing).

    In Cartesian coordinates (x, y) where w = Z_t - Λ(t) = x + iy:
        dx/dt = ax - ωy - b(x²+y²)x + (x²+y²)²x + D₂ dW_x/dt
        dy/dt = ay + ωx - b(x²+y²)y + (x²+y²)²y + D₂ dW_y/dt
    """

    def __init__(self, config: BautinConfig):
        self._config = config

    def get_parameters(self):
        """Get the parameters of the model."""
        return {
            "r": self._config.r,
            "lambda_max": self._config.lambda_max,
            "a": self._config.a,
            "omega": self._config.omega,
            "b": self._config.b,
            "D2": self._config.D2,
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
        config: BautinConfig,
    ):
        """Compute the Λ(t) schedule (time-varying forcing)."""
        # Λ(t) = 0.5Λ_max[tanh(0.5Λ_max r t) + 1]
        return 0.5 * config.lambda_max * (
            jnp.tanh(0.5 * config.lambda_max * config.r * t) + 1.0
        )

    def _sample_initial_state(
        self,
        key: jax.random.PRNGKey,
        config: BautinConfig,
    ):
        """Sample an initial state in Cartesian coordinates (x, y)."""
        key_x, key_y = jax.random.split(key)
        x = float(
            jax.random.uniform(
                key_x,
                (),
                minval=config.initial_state_min,
                maxval=config.initial_state_max,
            )
        )
        y = float(
            jax.random.uniform(
                key_y,
                (),
                minval=config.initial_state_min,
                maxval=config.initial_state_max,
            )
        )
        return x, y

    def _simulate_path(
        self,
        key: jax.random.PRNGKey,
        lambda_t: jnp.ndarray,
        dt: float,
        noise_scale: float,
        initial_x: float,
        initial_y: float,
        noise: Optional[jnp.ndarray] = None,
    ):
        """Stochastic integration in Cartesian coordinates.

        Implements: dZ_t = (a + iω)[Z_t - Λ(t)]dt - b[Z_t - Λ(t)]²[Z_t - Λ(t)]dt
                    + [Z_t - Λ(t)]⁴[Z_t - Λ(t)]dt + D₂dW₂t
        where w = Z_t - Λ(t) = x + iy
        """
        if noise is None:
            key, subkey = jax.random.split(key)
            noise_x = jax.random.normal(key, shape=(lambda_t.shape[0] - 1,))
            noise_y = jax.random.normal(subkey, shape=(lambda_t.shape[0] - 1,))
        else:
            noise = jnp.asarray(noise)
            noise_x = noise[:, 0]
            noise_y = noise[:, 1]

        def step(state, inputs):
            lambda_val, noise_x_t, noise_y_t = inputs
            x, y = state

            # Compute w = Z_t - Λ(t) = (x, y) - (Λ(t), 0) = (x - Λ(t), y)
            w_x = x - lambda_val
            w_y = y

            # Compute |w|² = w_x² + w_y²
            w_squared = w_x**2 + w_y**2
            w_fourth = w_squared**2

            # dx/dt = ax - ωy - b(x²+y²)x + (x²+y²)²x
            # where (x, y) here refers to w = (w_x, w_y)
            dx_dt = (
                self._config.a * w_x
                - self._config.omega * w_y
                - self._config.b * w_squared * w_x
                + w_fourth * w_x
            )
            x_next = x + dx_dt * dt + noise_scale * noise_x_t

            # dy/dt = ay + ωx - b(x²+y²)y + (x²+y²)²y
            dy_dt = (
                self._config.a * w_y
                + self._config.omega * w_x
                - self._config.b * w_squared * w_y
                + w_fourth * w_y
            )
            y_next = y + dy_dt * dt + noise_scale * noise_y_t

            return (x_next, y_next), (x_next, y_next)

        initial_state = (initial_x, initial_y)
        _, states = lax.scan(step, initial_state, (lambda_t[:-1], noise_x, noise_y))

        # Build full trajectory including initial state
        x_traj = jnp.concatenate([jnp.array([initial_x]), states[0]], axis=0)
        y_traj = jnp.concatenate([jnp.array([initial_y]), states[1]], axis=0)

        return jnp.stack([x_traj, y_traj], axis=-1)

    def simulate(
        self,
        key: jax.random.PRNGKey,
        total_len: int,
        sigma_tilde: float = 0.01,
        initial_state: Optional[jnp.ndarray] = None,
        noise: Optional[jnp.ndarray] = None,
        null_simulation: bool = False,
    ):
        """Simulate the Bautin system.

        Time range is fixed to [t_start, t_end] from config (paper: -15 to 15).
        total_len is ignored. The window length sets the ramp speed: wider (e.g. -20 to 20)
        → slower ramp → tipping tends to occur later; narrower → earlier tipping.
        For null_simulation=True, same ramp shape but with r at 50% (slower rate; visible in plot).
        """
        key_noise, key_ic = jax.random.split(key)

        # Setup time and lambda schedule (fixed -15 to 15 to match paper; do not use total_len)
        time = jnp.arange(self._config.t_start, self._config.t_end, self._config.dt)
        if null_simulation:
            config_null = replace(self._config, r=0.5 * self._config.r)
            lambda_t = self._lambda_schedule(time, config_null)
        else:
            lambda_t = self._lambda_schedule(time, self._config)

        # Randomly sample initial state if not provided
        if initial_state is None:
            initial_x, initial_y = self._sample_initial_state(key_ic, self._config)
        else:
            initial_state = jnp.asarray(initial_state)
            initial_x, initial_y = initial_state[0], initial_state[1]

        # Configure noise scale: D2 * sigma_tilde * sqrt(dt) for Euler-Maruyama
        noise_scale = self._config.D2 * sigma_tilde * jnp.sqrt(self._config.dt)

        # Simulate trajectory
        trajectory = self._simulate_path(
            key_noise,
            lambda_t,
            self._config.dt,
            noise_scale,
            initial_x,
            initial_y,
            noise=noise,
        )

        # Return lambda_t as param_values for consistency with other models
        # Return noise scale (like saddle_node) for consistency
        return time, trajectory, lambda_t, noise_scale

    def is_tipped(
        self,
        trajectory: jnp.ndarray,
        **kwargs,
    ):
        """Check if the trajectory is tipped (radius exceeds threshold)."""
        # Compute radius from Cartesian coordinates
        radius = jnp.sqrt(trajectory[:, 0] ** 2 + trajectory[:, 1] ** 2)
        return bool(jnp.any(radius > self._config.tip_threshold))

    def locate_transition(
        self,
        trajectory: jnp.ndarray,
        null_simulation: bool,
        **kwargs,
    ):
        """Locate the transition index."""
        if null_simulation:
            return int(trajectory.shape[0] // 2)

        # Compute radius and check if it exceeds threshold
        radius = jnp.sqrt(trajectory[:, 0] ** 2 + trajectory[:, 1] ** 2)
        crossed = jnp.where(radius > self._config.tip_threshold, 1, 0)
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
