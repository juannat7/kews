"""Generate the tipping and non-tipping trajectories used in the paper."""

import jax

from .bifurcation.bifurcation_simulator import simulate_real_bifurcation_trajectory
from .rate.rate_simulator import simulate_rate_trajectory
from .registry import SYSTEM_REGISTRY, SystemType


def generate(
    key,
    tipping_type,
    null_simulation=False,
    num_timesteps=500,
    steps_before_transition=1,
    buffer_size=1000,
    sigma_tilde=0.01,
    max_attempts=100,
    **generate_kwargs,
):
    """Return full trajectory, pre-transition segment, and simulation metadata.

    ``tipping_type`` selects one of the six entries in ``SYSTEM_REGISTRY``.
    ``num_timesteps`` is the extracted segment length. ``buffer_size`` adds
    pre-transition history without changing the parameter ramp rate, and
    ``steps_before_transition`` leaves a gap before the detected transition.
    Both trajectory results expose ``time``, ``trajectory``, ``param_values``,
    ``sigma``, and ``metadata``. Failed samples are retried with fresh keys;
    failure after ``max_attempts`` raises ``RuntimeError``.
    """
    if tipping_type not in SYSTEM_REGISTRY:
        raise ValueError(f"Unknown system {tipping_type!r}; choose from {sorted(SYSTEM_REGISTRY)}")
    if num_timesteps < 1 or max_attempts < 1 or steps_before_transition < 0 or buffer_size < 0:
        raise ValueError("Lengths and attempts must be positive; gap and buffer must be nonnegative")
    if steps_before_transition > buffer_size:
        raise ValueError("steps_before_transition exceeds buffer_size")

    spec = SYSTEM_REGISTRY[tipping_type]
    is_rate = spec.system_type == SystemType.RATE
    simulator = simulate_rate_trajectory if is_rate else simulate_real_bifurcation_trajectory
    system_name = spec.rate_system if is_rate else spec.bifurcation_system
    for attempt in range(max_attempts):
        # Keep this split order fixed: it determines each seeded simulation and retry.
        key, subkey = jax.random.split(key)
        result = simulator(
            subkey,
            series_len=num_timesteps,
            steps_before_transition=steps_before_transition,
            buffer_size=buffer_size,
            null_simulation=null_simulation,
            sigma_tilde=sigma_tilde,
            system_name=system_name,
            **generate_kwargs,
        )
        if result is not None:
            full, part = result
            info = dict(attempt=attempt + 1, system_name=tipping_type, type=spec.system_type.value)
            info.update(full.metadata)
            return full, part, info
    raise RuntimeError(f"Could not generate {tipping_type} trajectory after {max_attempts} attempts")
