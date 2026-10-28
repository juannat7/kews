"""
AMOC 2-box model, rate-induced tipping (Ritchie et al. 2023; Alkhayuon et al. 2019).

H(t) = ramp then plateau (forcing_ramp).
Tip threshold: transition during ramp phase (t < 0).
"""
from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
from jax import lax

from ..types import BoxModel
from ..utils import extract_segment, register_dataclass_pytree

# 2x CO2 parameters from BoxModel_2DH_IVP.m (rtippingreview)
_S0 = 0.035
_alpha = 0.12
_beta = 790.0
_Y = 100 * 3.15e7
_VN = 0.3683e17
_VT = 0.5418e17
_VS = 0.6097e17
_VIP = 1.4860e17
_VB = 9.9250e17
_Ts = 7.919
_T0 = 3.87
_C = 4.4735e16
_lambda = 1.62e7
_gamma = 0.36
_mu = 22e-8
_KN = 1.762e6
_KS = 1.872e6
_SS = (0.034427 - _S0) * 100
_SB = (0.034538 - _S0) * 100

INIT_ON_STATE = jnp.array([0.0324, 0.1434])


def _fn_ft(H):
    """FN and FT in m^3/s."""
    FN = (0.486 + 0.1311 * H) * 1e6
    FT = (-0.997 + 0.6961 * H) * 1e6
    return FN, FT


def _amoc_strength_sv(SN):
    """AMOC strength Q in Sv."""
    return (
        _lambda
        * (_alpha * (_Ts - _T0) + _beta * (SN / 100.0 - _SS / 100.0))
        / (1.0 + _lambda * _alpha * _mu)
        * 1e-6
    )


def _boxmodel_drift(state: jnp.ndarray, H_val) -> jnp.ndarray:
    """Drift (RHS) of 2D AMOC box model. state = [SN, ST], H_val = freshwater hosing (Sv), can be JAX scalar."""
    SN, ST = state[0], state[1]
    SIP = 100.0 * (
        _C
        - (_VN * SN + _VT * ST + _VS * _SS + _VB * _SB) / 100.0
        - _S0 * (_VB + _VN + _VT + _VIP + _VS)
    ) / _VIP

    q_sv = _amoc_strength_sv(SN)
    q = q_sv * 1e6
    aq = jnp.abs(q)
    FN, FT = _fn_ft(H_val)

    z1_pos = (_Y / _VN) * (
        q * (ST / 100.0 - SN / 100.0)
        + _KN * (ST / 100.0 - SN / 100.0)
        - FN * _S0
    )
    z2_pos = (_Y / _VT) * (
        q * (_gamma * _SS / 100.0 + (1.0 - _gamma) * SIP / 100.0 - ST / 100.0)
        + _KS * (_SS / 100.0 - ST / 100.0)
        + _KN * (SN / 100.0 - ST / 100.0)
        - FT * _S0
    )
    z1_neg = (_Y / _VN) * (
        aq * (_SB / 100.0 - SN / 100.0)
        + _KN * (ST / 100.0 - SN / 100.0)
        - FN * _S0
    )
    z2_neg = (_Y / _VT) * (
        aq * (SN / 100.0 - ST / 100.0)
        + _KS * (_SS / 100.0 - ST / 100.0)
        + _KN * (SN / 100.0 - ST / 100.0)
        - FT * _S0
    )
    z1 = jnp.where(q >= 0, z1_pos, z1_neg)
    z2 = jnp.where(q >= 0, z2_pos, z2_neg)
    return jnp.array([z1, z2])


def _forcing_ramp(t: jnp.ndarray, r: float, dH: float, H0: float = 0.0) -> jnp.ndarray:
    """H(t) = H0 + dH/cosh(r*t) for t < 0, H0 + dH for t >= 0 (ramp then plateau)."""
    out = jnp.where(t < 0, H0 + dH / jnp.cosh(r * t), H0 + dH)
    return out


def _rk4_step(state: jnp.ndarray, H_cur: float, H_mid: float, H_next: float, dt: float) -> jnp.ndarray:
    """One step of 4th-order Runge-Kutta: d state/dt = drift(state, H(t)). Uses H at t, t+dt/2, t+dt.
    No clamp: reference (amoc_boxmodel + scipy) has SN go negative in collapsed state."""
    k1 = _boxmodel_drift(state, H_cur)
    k2 = _boxmodel_drift(state + 0.5 * dt * k1, H_mid)
    k3 = _boxmodel_drift(state + 0.5 * dt * k2, H_mid)
    k4 = _boxmodel_drift(state + dt * k3, H_next)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


@register_dataclass_pytree
@dataclass(frozen=True)
class AMOCRateConfig:
    """Configuration for AMOC 2-box model (rate-induced)."""

    H0: float = 0.0
    dH: float = 0.365  # From repo (tipping_movie_AMOC.m); H_plateau = 0.365 < H_fold
    r_slow: float = 0.005
    r_fast: float = 0.017
    sigma_SN: float = 1e-4
    sigma_ST: float = 1e-4
    sigma_ic: float = 0.01
    dt: float = 5.0  # Output time step (years); we subsample to this from the integration grid
    dt_inner: float = 0.2  # Fixed-step RK4 integration step (years). Not adaptive RK45; small fixed step then subsample is standard.
    t_start: float = -2505.0  # Need ≥501 points before t=0 for extract_segment(series_len=500, steps_before=1)
    t_end: float = 3200.0
    # Tipped when Q (AMOC strength in Sv) < 0 (collapsed / reversed).
    Q_collapse_threshold: float = 0.0

    @classmethod
    def default(cls) -> "AMOCRateConfig":
        return cls()


class AMOCRateModel(BoxModel):
    """AMOC 2-box model, rate-induced. H(t) = ramp then plateau (peak at t=0). Transition = when t crosses 0 (H peaks/plateaus). is_tipped = Q collapsed (below physical threshold)."""

    def __init__(self, config: AMOCRateConfig):
        self._config = config

    def get_parameters(self):
        return {
            "H0": self._config.H0,
            "dH": self._config.dH,
            "r_slow": self._config.r_slow,
            "r_fast": self._config.r_fast,
            "sigma_SN": self._config.sigma_SN,
            "sigma_ST": self._config.sigma_ST,
            "sigma_ic": self._config.sigma_ic,
            "dt": self._config.dt,
            "t_start": self._config.t_start,
            "t_end": self._config.t_end,
            "Q_collapse_threshold": self._config.Q_collapse_threshold,
        }

    def _H_schedule(self, t: jnp.ndarray, null_simulation: bool) -> jnp.ndarray:
        """H(t) = forcing_ramp(t, r, dH, H0). null_simulation=True -> r_slow (no tip); False -> r_fast (tip)."""
        r = self._config.r_slow if null_simulation else self._config.r_fast
        return _forcing_ramp(t, r, self._config.dH, self._config.H0)

    def _sample_initial_state(self, key: jax.random.PRNGKey) -> jnp.ndarray:
        """Random initial state: Normal(INIT_ON_STATE, sigma_ic)."""
        noise = jax.random.normal(key, (2,))
        x0 = INIT_ON_STATE + self._config.sigma_ic * noise
        return jnp.maximum(x0, 1e-6)

    def _simulate_path(
        self,
        H_cur: jnp.ndarray,
        H_mid: jnp.ndarray,
        H_next: jnp.ndarray,
        dt: float,
        initial_state: jnp.ndarray,
        noise_scale: jnp.ndarray,
        noise: jnp.ndarray,
    ) -> jnp.ndarray:
        """Single RK4 integrator: state_next = rk4_step(...) + noise_scale * noise_t. Use noise_scale=0 for deterministic."""
        def step(state, inputs):
            h_cur, h_mid, h_next, noise_t = inputs
            state_next = _rk4_step(state, h_cur, h_mid, h_next, dt) + noise_scale * noise_t
            return state_next, state_next

        _, states = lax.scan(
            step, initial_state, (H_cur, H_mid, H_next, noise)
        )
        return jnp.concatenate([initial_state[None, :], states], axis=0)

    def simulate(
        self,
        key: jax.random.PRNGKey,
        total_len: int,
        sigma_tilde: float = 0.01,
        initial_state: Optional[jnp.ndarray] = None,
        noise: Optional[jnp.ndarray] = None,
        null_simulation: bool = False,
    ):
        """Simulate rate-induced AMOC. Fixed time range [t_start, t_end]; total_len ignored. Both branches use RK4."""
        key_noise, _ = jax.random.split(key)

        dt_out = self._config.dt
        dt_inner = self._config.dt_inner
        time_inner = jnp.arange(self._config.t_start, self._config.t_end, dt_inner)
        H_inner = self._H_schedule(time_inner, null_simulation)
        t_mid = time_inner[:-1] + 0.5 * dt_inner
        r = self._config.r_slow if null_simulation else self._config.r_fast
        H_mid = _forcing_ramp(t_mid, r, self._config.dH, self._config.H0)
        H_cur, H_next = H_inner[:-1], H_inner[1:]

        if initial_state is None:
            initial_state = INIT_ON_STATE
        else:
            initial_state = jnp.asarray(initial_state)

        deterministic = sigma_tilde <= 1e-9
        n_steps = H_cur.shape[0]
        if deterministic:
            noise_scale = jnp.zeros(2)
            noise_arr = jnp.zeros((n_steps, 2))
        else:
            noise_scale = jnp.array([
                self._config.sigma_SN * jnp.sqrt(dt_inner) * sigma_tilde,
                self._config.sigma_ST * jnp.sqrt(dt_inner) * sigma_tilde,
            ])
            noise_arr = jax.random.normal(key_noise, (n_steps, 2)) if noise is None else jnp.asarray(noise)
        path_inner = self._simulate_path(
            H_cur, H_mid, H_next, dt_inner, initial_state, noise_scale, noise_arr
        )

        step = max(1, int(dt_out / dt_inner))
        time = time_inner[::step]
        path = path_inner[::step]
        H_t = H_inner[::step]

        Q_t = _amoc_strength_sv(path[:, 0])
        trajectory = Q_t  # (T,)
        sigma_float = float(jnp.max(noise_scale))
        return time, trajectory, H_t, sigma_float, path

    def is_tipped(
        self,
        trajectory: jnp.ndarray,
        time=None,
        param_values=None,
        **kwargs,
    ):
        """True if Q collapsed (Q < 0, i.e. below Q_collapse_threshold)."""
        return bool(jnp.any(trajectory < self._config.Q_collapse_threshold))

    def locate_transition(
        self,
        trajectory: jnp.ndarray,
        null_simulation: bool,
        time=None,
        param_values=None,
        **kwargs,
    ):
        """Transition = when t crosses 0 (H peaks and plateaus). Returns first index where time >= 0. Caller must pass time."""
        if null_simulation:
            return int(trajectory.shape[0] // 2)
        if time is None:
            return None
        time = jnp.asarray(time)
        idx = jnp.searchsorted(time, 0.0, side="left")
        idx = jnp.minimum(idx, time.shape[0] - 1)
        return int(idx)

    def extract_segment(
        self,
        time: jnp.ndarray,
        trajectory: jnp.ndarray,
        param_values: jnp.ndarray,
        transition_idx: int,
        steps_before_transition: int,
        series_len: int,
    ):
        time_seg, traj_seg, param_seg, trans_idx_out, success = extract_segment(
            time,
            trajectory,
            param_values,
            transition_idx,
            steps_before_transition,
            series_len,
        )
        return time_seg, traj_seg, param_seg, trans_idx_out, success
