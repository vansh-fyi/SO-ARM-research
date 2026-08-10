"""State-based determinism replay verification for SOARM datasets (DATA-02).

Two verification tiers (D-06's two-tier split), both built on the SAME
regeneration primitive ``hdf5_writer.py`` used to BUILD the dataset
(``ControlEnv.set_state`` / ``ControlEnv.set_init_state`` in
``env_wrapper.py``) — this module does not build a second replay engine, it
proves the round-trip:

  * ``verify_states_only`` — cheap, 100%-of-demos tier. No rendering: just
    ``sim.set_state_from_flattened`` + ``sim.forward()`` then re-read the
    flattened state back and compare to the recorded value.
  * ``verify_full_obs_regeneration`` — expensive, sampled tier (default ~10%
    of states, minimum 5, always including at least the first state of every
    demo). Regenerates real image + proprio observations via
    ``set_init_state`` and compares them against the recorded ``obs/`` arrays.

Action-replay (``env.step(action)`` loops) is explicitly NOT used here — per
LIBERO issue #16, MuJoCo's contact solver is not bit-deterministic under
action replay, which is not what DATA-02 asks for (state-based determinism).

Both entry points accept either a path string (opened/closed internally) or
an already-open ``h5py.File``.
"""

import math
import os
import re

# Set the MuJoCo GL backend before the import that triggers MuJoCo (matches
# raw_recorder.py / hdf5_writer.py / collector.py's established convention).
os.environ.setdefault("MUJOCO_GL", "glfw")

import h5py
import numpy as np

from ..envs import OffScreenRenderEnv
from .hdf5_writer import OBS_KEY_MAPPING

_DEMO_RE = re.compile(r"^demo_(\d+)$")


def _demo_sort_key(name):
    """Sort demo_N groups numerically (demo_2 before demo_10), not lexically."""
    m = _DEMO_RE.match(name)
    return int(m.group(1)) if m else name


def _open_h5(hdf5_path):
    """Return (file_handle, should_close) for a path string or an open h5py.File."""
    if isinstance(hdf5_path, h5py.File):
        return hdf5_path, False
    return h5py.File(hdf5_path, "r"), True


def _demo_names(grp):
    return sorted(
        (k for k in grp.keys() if k.startswith("demo_")), key=_demo_sort_key
    )


def _build_env(bddl_file_name, robots=("Soarm101",), camera_size=128):
    """One OffScreenRenderEnv per unique bddl_file_name (this project's
    per-file convention: one bddl_file_name attr per HDF5 file)."""
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file_name,
        robots=list(robots),
        camera_heights=camera_size,
        camera_widths=camera_size,
        has_renderer=False,
        has_offscreen_renderer=True,
    )
    env.reset()
    return env


def verify_states_only(hdf5_path):
    """Cheap, 100%-of-demos state-setting round-trip check (D-06, tier 1).

    For every recorded ``demo_N/states`` row: ``sim.set_state_from_flattened``
    (via ``ControlEnv.set_state``) + ``sim.forward()``, then re-read the
    flattened sim state and assert it exactly equals the state that was just
    set. No rendering, no observation regeneration — pure state-vector math.

    Args:
        hdf5_path (str | h5py.File): Dataset path, or an already-open file.

    Returns:
        dict: {"total_demos": N, "total_states": M, "passed": M}.

    Raises:
        AssertionError: on the first non-finite recorded state value, or the
            first state that fails to round-trip exactly, naming the demo
            and step index.
    """
    f, should_close = _open_h5(hdf5_path)
    env = None
    try:
        grp = f["data"]
        bddl_file_name = grp.attrs["bddl_file_name"]
        env = _build_env(bddl_file_name)

        demo_names = _demo_names(grp)
        total_states = 0
        passed = 0
        for demo_name in demo_names:
            states = grp[demo_name]["states"][:]
            for step_index, state in enumerate(states):
                total_states += 1
                if not np.all(np.isfinite(state)):
                    raise AssertionError(
                        "verify_states_only: non-finite recorded state in "
                        f"{demo_name} step {step_index}"
                    )
                env.set_state(state)
                env.sim.forward()
                round_tripped = env.get_sim_state()
                if not np.allclose(round_tripped, state, atol=0.0):
                    raise AssertionError(
                        "verify_states_only: state round-trip mismatch in "
                        f"{demo_name} step {step_index}"
                    )
                passed += 1

        return {
            "total_demos": len(demo_names),
            "total_states": total_states,
            "passed": passed,
        }
    finally:
        if env is not None:
            env.close()
        if should_close:
            f.close()


def _select_sample(demo_names, per_demo_lengths, sample_size):
    """Deterministic sample: always include (demo, 0) for every demo, then
    fill the remaining budget with an evenly-spaced deterministic sweep over
    the rest of the (demo, step) universe (never a naive prefix, which would
    skew toward the earliest demos)."""
    must_include = [(d, 0) for d in demo_names if per_demo_lengths[d] > 0]
    selected = list(must_include)
    selected_set = set(selected)

    remaining_budget = max(0, sample_size - len(selected))
    if remaining_budget > 0:
        pool = [
            (d, i)
            for d in demo_names
            for i in range(per_demo_lengths[d])
            if (d, i) not in selected_set
        ]
        if pool:
            take = min(remaining_budget, len(pool))
            stride = max(1, len(pool) // take)
            selected.extend(pool[::stride][:take])

    return selected


def verify_full_obs_regeneration(hdf5_path, sample_size=None):
    """Sampled, full observation-regeneration determinism check (D-06, tier 2).

    Catches obs-extraction bugs the cheap states-only tier can't see (e.g. a
    wrong OBS_KEY_MAPPING entry or a stale render). For each sampled
    (demo, step): call ``set_init_state(recorded_state)`` (the same primitive
    ``hdf5_writer.py`` used to build the dataset), rename via
    ``OBS_KEY_MAPPING``, and assert exact equality against the recorded obs
    array for all 4 keys.

    Args:
        hdf5_path (str | h5py.File): Dataset path, or an already-open file.
        sample_size (int | None): Total number of (demo, step) samples to
            check. Defaults to ``max(5, ceil(0.10 * total_states))``. The
            first state of every demo is ALWAYS included regardless of
            ``sample_size`` (>=1/demo guarantee), so the effective sample can
            exceed a ``sample_size`` smaller than the demo count.

    Returns:
        dict: {"sampled_states": K, "passed": K}.

    Raises:
        AssertionError: on the first sampled state whose regenerated
            observation diverges from the recorded one, naming
            (demo_name, step_index, key).
    """
    f, should_close = _open_h5(hdf5_path)
    env = None
    try:
        grp = f["data"]
        bddl_file_name = grp.attrs["bddl_file_name"]
        demo_names = _demo_names(grp)
        per_demo_lengths = {d: grp[d]["states"].shape[0] for d in demo_names}
        total_states = sum(per_demo_lengths.values())

        if sample_size is None:
            sample_size = max(5, math.ceil(0.10 * total_states))

        selected = _select_sample(demo_names, per_demo_lengths, sample_size)

        env = _build_env(bddl_file_name)

        sampled = 0
        passed = 0
        for demo_name, step_index in selected:
            sampled += 1
            state = grp[demo_name]["states"][step_index]
            obs = env.set_init_state(state)
            for schema_key, robosuite_key in OBS_KEY_MAPPING.items():
                recorded = grp[demo_name]["obs"][schema_key][step_index]
                regenerated = obs[robosuite_key]
                if not np.allclose(regenerated, recorded):
                    raise AssertionError(
                        "verify_full_obs_regeneration: mismatch in "
                        f"({demo_name}, {step_index}, {schema_key})"
                    )
            passed += 1

        return {"sampled_states": sampled, "passed": passed}
    finally:
        if env is not None:
            env.close()
        if should_close:
            f.close()
