from dataclasses import asdict
from typing import Any, Dict

import jax

from ..types import TrajectoryResult
from ..registry import get_rate_system_spec
from ..utils import calculate_total_len


def simulate_rate_trajectory(
    key: jax.random.PRNGKey,
    system_name: str,
    series_len: int = 500,
    steps_before_transition: int = 1,
    buffer_size: int = 1000,
    sigma_tilde: float = 0.01,
    null_simulation: bool = False,
    param_end_multiplier: float = 1.1,
    **generate_kwargs
):
    """Simulate a single rate-induced tipping trajectory.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for reproducibility
    series_len : int, default=500
        Desired output trajectory length
    steps_before_transition : int, default=1
        Number of samples to stay before the detected transition (>=0).
    buffer_size : int, default=1000
        Extra samples simulated prior to the transition to ensure enough
        pre-transition context. Keep this constant to preserve the
        bifurcation parameter ramp rate.
    null_simulation : bool, default=False
        If True, simulate null (non-tipping) trajectory
    system_name : str
        Internal name of the rate system selected by generate().
        Options: 'bautin', 'saddle_node', 'amoc'
    param_end_multiplier : float, default=1.1
        Multiplier for extending simulation time beyond transition
    **generate_kwargs : additional arguments
        Passed to the model.simulate method (dt_sample, dt_integrate, etc.)

    Returns
    -------
    full_trajectory : experiments.systems.types.TrajectoryResult
        Full trajectory with metadata
    part_trajectory : experiments.systems.types.TrajectoryResult
        Part trajectory with metadata

    Returns None if the simulated trajectory does not match the requested
    tipping/null behaviour or lacks sufficient pre-transition context.
    """

    # Preserve the simulation subkey used by the paper reference.
    _, key_sim = jax.random.split(key)


    # Calculate total simulation length (extended if param_end_multiplier > 1.0)
    total_len = calculate_total_len(series_len, buffer_size, param_end_multiplier)

    # Select, setup and initialize model using unified registry
    system_spec = get_rate_system_spec(system_name)

    config = system_spec.default_config()
    model = system_spec.create_model(config)

    # Simulate (saddle_node, bautin use null_simulation: 50% rate in schedule)
    sim_kwargs = {"total_len": total_len, "sigma_tilde": sigma_tilde}
    if system_spec.rate_system in ("saddle_node", "bautin", "amoc"):
        sim_kwargs["null_simulation"] = null_simulation
    sim_ret = model.simulate(key_sim, **sim_kwargs)
    time, trajectory, param_values, sigma = sim_ret[0], sim_ret[1], sim_ret[2], sim_ret[3]
    state_trajectory = sim_ret[4] if len(sim_ret) > 4 else None

    # Check if the trajectory is tipped (pass time/param_values for models that need them, e.g. AMOC)
    is_tipped = model.is_tipped(
        trajectory,
        time=time,
        param_values=param_values,
    )
    if is_tipped == null_simulation:
        return None

    # Locate transition index (transition = when t crosses 0 for rate-induced AMOC)
    trans_idx = model.locate_transition(
        trajectory,
        null_simulation=null_simulation,
        time=time,
        param_values=param_values,
    )

    if trans_idx is None:
        return None

    # Extract segment
    time_seg, traj_seg, param_seg, trans_idx_out, success = model.extract_segment(
        time,
        trajectory,
        param_values,
        trans_idx,
        steps_before_transition=steps_before_transition,
        series_len=series_len,
    )

    if not success:
        return None

    # Return results
    metadata: Dict[str, Any] = {
        "label": "null" if null_simulation else "rate",
        "rate_system": system_spec.rate_system,
        "steps_before_transition": steps_before_transition,
        "system_config": asdict(config),
        "transition_index": int(trans_idx_out),
        "param_value_transition": float(param_values[min(trans_idx_out, param_values.shape[0] - 1)])
    }
    if state_trajectory is not None:
        metadata["state_trajectory"] = state_trajectory

    full_trajectory = TrajectoryResult(
        time=time,
        trajectory=trajectory,
        param_values=param_values,
        sigma=sigma,
        metadata=metadata
    )

    part_trajectory = TrajectoryResult(
        time=time_seg,
        trajectory=traj_seg,
        param_values=param_seg,
        sigma=sigma,
        metadata=metadata
    )

    return (full_trajectory, part_trajectory)
