from dataclasses import dataclass
from typing import Optional, Literal

import jax
import jax.numpy as jnp
from jax import lax

from ..types import BoxModel
from ..utils import extract_segment, register_dataclass_pytree


@register_dataclass_pytree
@dataclass(frozen=True)
class RosenzweigMacArthurConfig:
    """Configuration for Rosenzweig-MacArthur consumer-resource model.
    
    Based on Thomas Bury et al. (PNAS, 2021):
    dx/dt = rx(1 - x/k) - axy/(1 + ahx) + σ₁ξ₁(t)
    dy/dt = eaxy/(1 + ahx) - my + σ₂ξ₂(t)
    
    where:
    - x: resource (prey) population
    - y: consumer (predator) population
    - r: intrinsic per capita growth rate of resource
    - k: carrying capacity
    - a: attack rate of consumer (bifurcation parameter)
    - e: conversion factor
    - h: handling time
    - m: per capita consumer mortality rate
    - σ₁, σ₂: noise amplitudes
    - ξ₁(t), ξ₂(t): independent Gaussian white noise processes
    
    Bifurcation points (for default parameters):
    - Transcritical bifurcation at a = 5.60
    - Hopf bifurcation at a = 15.69
    """

    # Model parameters (from Bury et al. 2021)
    r: float = 4.0       # Intrinsic per capita growth rate of resource
    k: float = 1.7       # Carrying capacity
    e: float = 0.5       # Conversion factor
    h: float = 0.15      # Handling time
    m: float = 2.0       # Per capita consumer mortality rate
    
    # Noise parameters
    sigma1: float = 0.01  # Noise amplitude for x (resource)
    sigma2: float = 0.01  # Noise amplitude for y (consumer)
    
    # Bifurcation type: "transcritical" or "hopf"
    bifurcation_type: str = "transcritical"
    
    # Bifurcation parameter ranges (attack rate a)
    # Transcritical: null at a=2, forced a ∈ [2, 6], bifurcation at a=5.60
    a_transcritical_null: float = 2.0
    a_transcritical_min: float = 2.0
    a_transcritical_max: float = 6.0
    a_transcritical_bifurcation: float = 5.60
    
    # Hopf: null at a=12, forced a ∈ [12, 16], bifurcation at a=15.69
    a_hopf_null: float = 12.0
    a_hopf_min: float = 12.0
    a_hopf_max: float = 17.0
    a_hopf_bifurcation: float = 15.69
    
    # Simulation parameters
    dt: float = 0.01      # Time step (Euler-Maruyama)
    t_start: float = 0.0
    t_end: float = 100.0  # Default time range

    @classmethod
    def default(cls, bifurcation_type: str = "transcritical") -> "RosenzweigMacArthurConfig":
        """Return default configuration for specified bifurcation type."""
        return cls(bifurcation_type=bifurcation_type)
    
    @classmethod
    def transcritical(cls) -> "RosenzweigMacArthurConfig":
        """Return configuration for transcritical bifurcation."""
        return cls(bifurcation_type="transcritical")
    
    @classmethod
    def hopf(cls) -> "RosenzweigMacArthurConfig":
        """Return configuration for Hopf bifurcation."""
        return cls(bifurcation_type="hopf")


class RosenzweigMacArthurModel(BoxModel):
    """BoxModel abstraction for Rosenzweig-MacArthur consumer-resource model.
    
    Implements the stochastic consumer-resource model:
    dx/dt = rx(1 - x/k) - axy/(1 + ahx) + σ₁ξ₁(t)
    dy/dt = eaxy/(1 + ahx) - my + σ₂ξ₂(t)
    
    where a (attack rate) is the bifurcation parameter that varies linearly.
    
    This model exhibits two bifurcations:
    - Transcritical bifurcation at a = 5.60 (consumer invasion)
    - Hopf bifurcation at a = 15.69 (onset of predator-prey cycles)
    """

    def __init__(self, config: RosenzweigMacArthurConfig):
        self._config = config
        
    @property
    def bifurcation_type(self) -> str:
        """Get the bifurcation type being simulated."""
        return self._config.bifurcation_type

    def get_parameters(self):
        """Get the parameters of the model."""
        return {
            "r": self._config.r,
            "k": self._config.k,
            "e": self._config.e,
            "h": self._config.h,
            "m": self._config.m,
            "sigma1": self._config.sigma1,
            "sigma2": self._config.sigma2,
            "bifurcation_type": self._config.bifurcation_type,
            "a_transcritical_null": self._config.a_transcritical_null,
            "a_transcritical_min": self._config.a_transcritical_min,
            "a_transcritical_max": self._config.a_transcritical_max,
            "a_transcritical_bifurcation": self._config.a_transcritical_bifurcation,
            "a_hopf_null": self._config.a_hopf_null,
            "a_hopf_min": self._config.a_hopf_min,
            "a_hopf_max": self._config.a_hopf_max,
            "a_hopf_bifurcation": self._config.a_hopf_bifurcation,
            "dt": self._config.dt,
            "t_start": self._config.t_start,
            "t_end": self._config.t_end,
        }
    
    def _get_param_range(self) -> tuple:
        """Get the parameter range for the current bifurcation type."""
        if self._config.bifurcation_type == "transcritical":
            return (
                self._config.a_transcritical_null,
                self._config.a_transcritical_min,
                self._config.a_transcritical_max,
                self._config.a_transcritical_bifurcation,
            )
        else:  # hopf
            return (
                self._config.a_hopf_null,
                self._config.a_hopf_min,
                self._config.a_hopf_max,
                self._config.a_hopf_bifurcation,
            )

    def _a_schedule(
        self,
        t: jnp.ndarray,
        null_simulation: bool = False,
        param_end_multiplier: float = 1.1,
    ):
        """Compute the a(t) schedule (attack rate).
        
        For null simulations: same linear ramp as forced but clipped so the range from
        null_cap to critical equals the range from critical to a_max (symmetric about critical).
        For forced simulations:
          - a increases linearly from a_min to a_max
          - param_end_multiplier extends the time series beyond the bifurcation point
        """
        a_null, a_min, a_max, a_bif = self._get_param_range()
        t_total = t[-1] - t[0]
        t_to_bifurcation = t_total / param_end_multiplier
        t_rel = t - t[0]
        ramp = a_min + (t_rel / t_to_bifurcation) * (a_max - a_min)

        if null_simulation:
            range_past = a_max - a_bif
            cap = a_bif - range_past
            cap = jnp.maximum(cap, a_min)
            return jnp.minimum(ramp, cap)
        else:
            a_values = jnp.where(
                t_rel <= t_to_bifurcation,
                ramp,
                a_max
            )
            return a_values

    def _compute_equilibrium(self, a: float) -> tuple:
        """Compute the equilibrium state for given attack rate a.
        
        For the transcritical bifurcation region (a < 5.60):
        - Below transcritical: consumer-free equilibrium (x*, 0)
        - Above transcritical: coexistence equilibrium
        
        For the Hopf bifurcation region (a > 5.60):
        - Stable coexistence equilibrium (before Hopf)
        - Limit cycle behavior (after Hopf at a = 15.69)
        """
        r, k, e, h, m = (
            self._config.r,
            self._config.k,
            self._config.e,
            self._config.h,
            self._config.m,
        )
        
        # Transcritical threshold
        a_tc = self._config.a_transcritical_bifurcation
        
        if a < a_tc:
            # Consumer-free equilibrium: x = k, y = 0
            return k, 0.0
        else:
            # Coexistence equilibrium
            # From dy/dt = 0: eax/(1+ahx) = m
            # Solving for x: x* = m / (a(e - mh))
            x_star = m / (a * (e - m * h))
            
            # From dx/dt = 0: r(1 - x/k) = ay/(1+ahx)
            # Solving for y: y* = r(1 - x*/k)(1 + ahx*) / a
            functional_response_denom = 1.0 + a * h * x_star
            y_star = r * (1.0 - x_star / k) * functional_response_denom / a
            
            return max(x_star, 0.01), max(y_star, 0.01)

    def _sample_initial_state(
        self,
        key: jax.random.PRNGKey,
        null_simulation: bool = False,
    ):
        """Sample an initial state (x, y).
        
        Samples near the equilibrium for the initial parameter value.
        For transcritical bifurcation, we need a positive y to allow
        consumer invasion (y=0 is always an equilibrium).
        """
        a_null, a_min, a_max, a_bif = self._get_param_range()
        
        key1, key2 = jax.random.split(key)
        
        if self._config.bifurcation_type == "transcritical":
            # For transcritical: start near carrying capacity for x
            # and with a small but significant consumer population
            x = float(jax.random.uniform(key1, (), minval=0.8, maxval=1.0) * self._config.k)
            
            # Consumer needs enough initial population to survive until a > bifurcation point
            # Start with ~5% of resource as consumer
            y = float(jax.random.uniform(key2, (), minval=0.02, maxval=0.08))
        else:
            # For Hopf: use coexistence equilibrium at ramp start (a_min; null now ramps from a_min too)
            a_init = a_min
            x_eq, y_eq = self._compute_equilibrium(a_init)
            
            # Add small noise around equilibrium
            x = float(jax.random.normal(key1, ()) * 0.05 + x_eq)
            y = float(jax.random.normal(key2, ()) * 0.05 + y_eq)
            
            x = max(x, 0.1)
            y = max(y, 0.01)
        
        return jnp.array([x, y])

    def _drift(self, state: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
        """Compute the drift terms for the consumer-resource model.
        
        dx/dt = rx(1 - x/k) - axy/(1 + ahx)
        dy/dt = eaxy/(1 + ahx) - my
        
        Parameters
        ----------
        state : [x, y] current state
        a : attack rate (bifurcation parameter)
        
        Returns
        -------
        drift : [dx/dt, dy/dt] (without noise)
        """
        x, y = state[0], state[1]
        r, k, e, h, m = (
            self._config.r,
            self._config.k,
            self._config.e,
            self._config.h,
            self._config.m,
        )
        
        # Functional response (Type II): axy / (1 + ahx)
        functional_response = a * x * y / (1.0 + a * h * x)
        
        # Resource dynamics: logistic growth - predation
        dx = r * x * (1.0 - x / k) - functional_response
        
        # Consumer dynamics: predation gain - mortality
        dy = e * functional_response - m * y
        
        return jnp.array([dx, dy])

    def _simulate_path(
        self,
        key: jax.random.PRNGKey,
        a_t: jnp.ndarray,
        dt: float,
        noise_scale: jnp.ndarray,
        initial_state: jnp.ndarray,
        noise: Optional[jnp.ndarray] = None,
    ):
        """Stochastic integration using Euler-Maruyama.
        
        dx = drift_x * dt + σ₁ * dW₁
        dy = drift_y * dt + σ₂ * dW₂
        """
        n_steps = a_t.shape[0] - 1
        
        if noise is None:
            noise = jax.random.normal(key, shape=(n_steps, 2))
        else:
            noise = jnp.asarray(noise)

        def step(state, inputs):
            a_val, noise_t = inputs
            
            # Drift
            drift = self._drift(state, a_val)
            
            # Euler-Maruyama: dx = drift * dt + noise_scale * dW
            state_next = state + drift * dt + noise_scale * noise_t
            
            # Ensure non-negative populations
            state_next = jnp.maximum(state_next, 0.0)
            
            return state_next, state_next

        _, states = lax.scan(step, initial_state, (a_t[:-1], noise))

        # Build full trajectory including initial state
        trajectory = jnp.concatenate([initial_state[None, :], states], axis=0)

        return trajectory

    def simulate(
        self,
        key: jax.random.PRNGKey,
        total_len: int,
        sigma_tilde: float = 0.01,
        null_simulation: bool = False,
        dt_sample: float = 1.0,
        param_end_multiplier: float = 1.1,
        initial_state: Optional[jnp.ndarray] = None,
        noise: Optional[jnp.ndarray] = None,
        **kwargs
    ):
        """Simulate Rosenzweig-MacArthur model.
        
        Parameters
        ----------
        key : JAX random key
        total_len : int
            Total number of output time points
        sigma_tilde : float, default=0.01
            Base noise amplitude
        null_simulation : bool, default=False
            If True, same linear ramp as forced but a capped so range to critical equals (a_max - critical) (null trajectory)
        dt_sample : float, default=1.0
            Sampling interval for output
        param_end_multiplier : float, default=1.1
            Multiplier for extending simulation time beyond transition
            
        Returns
        -------
        time : (total_len,) array
            Time points
        trajectory : (total_len, 2) array
            State trajectory [x, y]
        param_values : (total_len,) array
            Attack rate a(t) values
        noise_scale : (2,) array
            Effective noise scales used [σ₁, σ₂]
        """
        key_noise, key_ic = jax.random.split(key)

        # Setup time grid
        total_time = total_len * dt_sample
        n_steps = int(total_time / self._config.dt) + 1
        time_fine = jnp.linspace(self._config.t_start, total_time, n_steps)
        
        # Compute a schedule
        a_t = self._a_schedule(
            time_fine,
            null_simulation=null_simulation,
            param_end_multiplier=param_end_multiplier,
        )

        # Sample initial state if not provided
        if initial_state is None:
            initial_state = self._sample_initial_state(key_ic, null_simulation=null_simulation)
        else:
            initial_state = jnp.asarray(initial_state)

        # Configure noise scales: σ * sigma_tilde * sqrt(dt) for Euler-Maruyama
        noise_scale = jnp.array([
            sigma_tilde * jnp.sqrt(self._config.dt),
            sigma_tilde * jnp.sqrt(self._config.dt),
        ])

        # Simulate trajectory at fine time steps
        trajectory_fine = self._simulate_path(
            key_noise,
            a_t,
            self._config.dt,
            noise_scale,
            initial_state,
            noise=noise,
        )

        # Downsample to output rate
        sample_every = int(dt_sample / self._config.dt)
        indices = jnp.arange(0, n_steps, sample_every)
        
        time = time_fine[indices]
        trajectory = trajectory_fine[indices]
        param_values = a_t[indices]

        return time, trajectory, param_values, noise_scale

    def is_tipped(
        self,
        trajectory: jnp.ndarray,
        param_values: jnp.ndarray,
    ):
        """Check if the parameter crossed the bifurcation point."""
        _, _, _, a_bif = self._get_param_range()
        return float(jnp.max(param_values)) >= a_bif

    def locate_transition(
        self,
        trajectory: jnp.ndarray,
        null_simulation: bool,
        param_values: jnp.ndarray,
        param_end_multiplier: float = 1.1,
    ):
        """Locate transition index: midpoint for null, bifurcation crossing for forced.
        """
        if null_simulation:
            return len(param_values) // 2
        
        _, _, _, a_bif = self._get_param_range()
        
        crossed = param_values >= a_bif

        if bool(jnp.any(crossed)):
            return int(jnp.argmax(crossed))
        
        return len(param_values) - 1

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
