"""State-based determinism replay verification for SOARM datasets (DATA-02).

Two verification tiers (D-06's two-tier split), both built on the SAME
regeneration primitive ``hdf5_writer.py`` used to BUILD the dataset
(``ControlEnv.set_state`` / ``ControlEnv.set_init_state`` in
``env_wrapper.py``) — this module does not build a second replay engine, it
proves the round-trip:

  * ``verify_states_only`` — cheap, 100%-of-demos tier. No rendering: just
    ``sim.set_state_from_flattened`` + ``sim.forward()`` then re-read the
    flattened state back and compare to the recorded value.
  * ``verify_full_obs_regeneration`` — expensive, sampled tier (added in
    Task 2), default ~10% of states, minimum 5, always including at least the
    first state of every demo.

Action-replay (``env.step(action)`` loops) is explicitly NOT used here — per
LIBERO issue #16, MuJoCo's contact solver is not bit-deterministic under
action replay, which is not what DATA-02 asks for (state-based determinism).

Both entry points accept either a path string (opened/closed internally) or
an already-open ``h5py.File``.
"""

import os
import re

# Set the MuJoCo GL backend before the import that triggers MuJoCo (matches
# raw_recorder.py / hdf5_writer.py / collector.py's established convention).
os.environ.setdefault("MUJOCO_GL", "glfw")

import h5py
import numpy as np

from ..envs import OffScreenRenderEnv

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
