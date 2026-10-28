"""
AMOC 2-box model, bifurcation-induced tipping (Ritchie et al. 2023; Alkhayuon et al. 2019).

State (SN, ST); control H(t) (freshwater hosing).
Saddle-node (fold) at H_fold; ramp H(t) with cap for tipping vs null.
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
    """FN and FT in m^3/s (from BoxModel_2DH_IVP.m)."""
    FN = (0.486 + 0.1311 * H) * 1e6
    FT = (-0.997 + 0.6961 * H) * 1e6
    return FN, FT


def _amoc_strength_sv(SN):
    """AMOC strength Q in Sv (tipping_movie_AMOC.m)."""
    return (
        _lambda
        * (_alpha * (_Ts - _T0) + _beta * (SN / 100.0 - _SS / 100.0))
        / (1.0 + _lambda * _alpha * _mu)
        * 1e-6
    )


def _boxmodel_drift(state: jnp.ndarray, H_val) -> jnp.ndarray:
    """Drift (RHS) of 2D AMOC box model. H_val can be JAX scalar (for lax.scan)."""
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


def _forcing_bifurcation_ramp_cap(t: jnp.ndarray, H0: float, rate: float, H_cap: float) -> jnp.ndarray:
    """H(t) = min(H0 + rate*t, H_cap) — ramp then constant at H_cap (same as amoc_boxmodel.forcing_bifurcation_ramp_cap)."""
    return jnp.minimum(H0 + rate * t, H_cap)


def _bifurcation_ramp_cap_params(
    t_end: float,
    H0: float,
    H_fold: float,
    *,
    delta_cap: float = 0.05,
    tip_ramp_end: float = 0.5,
) -> tuple:
    """Rate and caps so H(t) reaches H_fold at t_end (like other box models: crossing at t_total/param_end_multiplier)."""
    rate = (H_fold - H0) / t_end
    H_cap_nontip = H_fold - delta_cap
    H_cap_tip = H_fold + delta_cap
    return rate, H_cap_nontip, H_cap_tip


@register_dataclass_pytree
@dataclass(frozen=True)
class AMOCConfig:
    """Configuration for AMOC 2-box model (bifurcation-induced)."""

    H0: float = 0.0
    H_fold: float = 0.4236
    delta_cap: float = 0.05
    tip_ramp_end: float = 0.5
    sigma_SN: float = 1e-4
    sigma_ST: float = 1e-4
    sigma_ic: float = 0.01  # std for random initial state (Normal about INIT_ON_STATE)
    dt: float = 5.0  # Output time step (years); same as rate AMOC
    t_start: float = 0.0
    t_end: float = 5200.0  # Total simulation time (years), match 04
    dt_inner: float = 0.2  # Fixed-step RK4 integration step (years). Small fixed step then subsample is standard.

    @classmethod
    def default(cls) -> "AMOCConfig":
        return cls()


class AMOCModel(BoxModel):
    """AMOC 2-box model, bifurcation-induced (H(t) ramp with fold at H_fold)."""

    def __init__(self, config: AMOCConfig):
        self._config = config

    def get_parameters(self):
        return {
            "H0": self._config.H0,
            "H_fold": self._config.H_fold,
            "delta_cap": self._config.delta_cap,
            "tip_ramp_end": self._config.tip_ramp_end,
            "sigma_SN": self._config.sigma_SN,
            "sigma_ST": self._config.sigma_ST,
            "sigma_ic": self._config.sigma_ic,
            "dt": self._config.dt,
            "t_start": self._config.t_start,
            "t_end": self._config.t_end,
        }

    def _H_schedule(
        self,
        t: jnp.ndarray,
        null_simulation: bool = False,
        param_end_multiplier: float = 1.1,
    ) -> jnp.ndarray:
        """H(t) = forcing_bifurcation_ramp_cap(t, H0, rate, H_cap)."""
        H0 = self._config.H0
        H_fold = self._config.H_fold
        delta_cap = self._config.delta_cap
        tip_ramp_end = self._config.tip_ramp_end
        t_total = t[-1] - t[0]
        t_end_eff = t_total / param_end_multiplier
        rate, H_cap_nontip, H_cap_tip = _bifurcation_ramp_cap_params(
            t_end_eff, H0, H_fold, delta_cap=delta_cap, tip_ramp_end=tip_ramp_end
        )
        H_cap = H_cap_nontip if null_simulation else H_cap_tip
        t_rel = t - t[0]  # time from 0 for forcing_bifurcation_ramp_cap
        return _forcing_bifurcation_ramp_cap(t_rel, H0, rate, H_cap)

    def _sample_initial_state(
        self,
        key: jax.random.PRNGKey,
        null_simulation: bool = False,
    ) -> jnp.ndarray:
        """Random initial state: Normal(INIT_ON_STATE, sigma_ic)."""
        key, _ = jax.random.split(key)
        noise = jax.random.normal(key, (2,))
        x0 = INIT_ON_STATE + self._config.sigma_ic * noise
        return jnp.maximum(x0, 1e-6)  # keep salinities positive

    def _drift(self, state: jnp.ndarray, H: jnp.ndarray) -> jnp.ndarray:
        return _boxmodel_drift(state, H)

    @staticmethod
    def _rk4_step(state: jnp.ndarray, H_cur: float, H_mid: float, H_next: float, dt: float) -> jnp.ndarray:
        """One RK4 step; no clamp so collapsed branch (SN < 0) is reachable."""
        k1 = _boxmodel_drift(state, H_cur)
        k2 = _boxmodel_drift(state + 0.5 * dt * k1, H_mid)
        k3 = _boxmodel_drift(state + 0.5 * dt * k2, H_mid)
        k4 = _boxmodel_drift(state + dt * k3, H_next)
        return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

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
            state_next = self._rk4_step(state, h_cur, h_mid, h_next, dt) + noise_scale * noise_t
            return state_next, state_next

        _, states = lax.scan(
            step, initial_state, (H_cur, H_mid, H_next, noise)
        )
        return jnp.concatenate([initial_state[None, :], states], axis=0)

    def simulate(
        self,
        key: jax.random.PRNGKey,
        total_len: int,
        sigma_tilde: float = 1.0,
        null_simulation: bool = False,
        dt_sample: float = 1.0,
        param_end_multiplier: float = 1.1,
        initial_state: Optional[jnp.ndarray] = None,
        noise: Optional[jnp.ndarray] = None,
        **kwargs
    ):
        key_noise, _ = jax.random.split(key)
        total_time = self._config.t_end
        dt_inner = self._config.dt_inner
        dt_out = self._config.dt
        # Single inner grid: RK4 for both branches (stochastic adds noise per step)
        time_inner = jnp.linspace(
            self._config.t_start, total_time, int(total_time / dt_inner) + 1
        )
        H_inner = self._H_schedule(
            time_inner, null_simulation=null_simulation, param_end_multiplier=param_end_multiplier
        )
        H_cur = H_inner[:-1]
        H_next = H_inner[1:]
        H_mid = (H_cur + H_next) / 2.0

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
        trajectory_fine = self._simulate_path(
            H_cur, H_mid, H_next, dt_inner, initial_state, noise_scale, noise_arr
        )

        sample_every = max(1, int(dt_out / dt_inner))
        n_inner = trajectory_fine.shape[0]
        indices = jnp.arange(0, n_inner, sample_every)
        time = time_inner[indices]
        path_sampled = trajectory_fine[indices]

        # Returned H(t) = forcing_bifurcation_ramp_cap at output times (so plot shows ramp-cap)
        t_end_eff = (time[-1] - time[0]) / param_end_multiplier
        rate, H_cap_nontip, H_cap_tip = _bifurcation_ramp_cap_params(
            t_end_eff, self._config.H0, self._config.H_fold,
            delta_cap=self._config.delta_cap, tip_ramp_end=self._config.tip_ramp_end,
        )
        H_cap = H_cap_nontip if null_simulation else H_cap_tip
        t_rel = time - time[0]
        param_values = _forcing_bifurcation_ramp_cap(t_rel, self._config.H0, rate, H_cap)

        # Output trajectory with both physical state (S_N, S_T) and AMOC strength Q(t).
        # Column convention:
        #   trajectory[:, 0] = Q(t)  (AMOC strength in Sv, for 1D EWS)
        #   trajectory[:, 1] = S_N(t)  (North Atlantic salinity)
        #   trajectory[:, 2] = S_T(t)  (Tropical Atlantic salinity)
        SN_sampled = path_sampled[:, 0]
        ST_sampled = path_sampled[:, 1]
        Q_t = _amoc_strength_sv(SN_sampled)
        trajectory = jnp.stack([Q_t, SN_sampled, ST_sampled], axis=1)

        sigma_float = float(jnp.max(noise_scale))
        return time, trajectory, param_values, sigma_float

    def is_tipped(
        self,
        trajectory: jnp.ndarray,
        param_values: jnp.ndarray,
    ):
        """True if H crossed the bifurcation point (max H >= H_fold)."""
        return float(jnp.max(param_values)) >= self._config.H_fold

    def locate_transition(
        self,
        trajectory: jnp.ndarray,
        null_simulation: bool,
        param_values: jnp.ndarray,
        param_end_multiplier: float = 1.1,
    ):
        if null_simulation:
            return len(param_values) // 2
        crossed = param_values >= self._config.H_fold
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
        time_seg, traj_seg, param_seg, trans_idx_out, success = extract_segment(
            time,
            trajectory,
            param_values,
            transition_idx,
            steps_before_transition,
            series_len,
        )
        return time_seg, traj_seg, param_seg, trans_idx_out, success
