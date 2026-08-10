"""Keyboard-only teleoperation collection loop for SOARM (Phase 4, DATA-04, D-02).

Adapted from ``LIBERO/scripts/collect_demonstration.py``'s
``collect_human_trajectory`` loop shape (reset -> render -> per-step
``input2action`` -> ``env.step`` -> success-hold state machine), but:

  * KEYBOARD ONLY — no 3D-mouse input-device branch at all (D-02 explicitly
    excludes that class of hardware dependency; this project has no such
    device and no intention of adding one).
  * Reuses this phase's own shared infrastructure verbatim —
    ``raw_recorder.build_recording_env`` (with ``has_renderer=True`` for the
    on-screen GLFW viewer, unlike ``collector.py``'s headless
    ``has_renderer=False``) and ``hdf5_writer.gather_demonstrations_as_hdf5``
    (the SAME writer function the scripted collector calls) — so schema
    convergence between the scripted and teleop collection paths is
    structural, not something to separately maintain.

This module intentionally mirrors ``collector.py``'s two-layer shape: a single
episode-driving loop (``run_teleop_episode``, analogous to
``run_scripted_episode``) plus a collection driver (``collect_teleop``,
analogous to ``collect_task``). The only real difference between the scripted
and teleop paths is the action SOURCE (human keyboard input vs. a computed
waypoint) — both converge on the same ``DataCollectionWrapper``-recorded
states/actions feeding the same HDF5 writer.

Requires a real on-screen display (local GLFW window) — this is NOT
Colab/headless-CI compatible, matching D-03/D-09's established local-only
convention for this phase's live collection step.
"""

import argparse
import os
import tempfile

# Set the MuJoCo GL backend before any import that triggers MuJoCo (mirrors
# raw_recorder.py / hdf5_writer.py / collector.py). setdefault so an
# externally-set osmesa value on Linux is not clobbered. Teleop specifically
# needs a real on-screen GLFW window (has_renderer=True), so this must run on
# a machine with a local display attached.
os.environ.setdefault("MUJOCO_GL", "glfw")

from robosuite.devices import Keyboard  # noqa: E402  (import order matches MUJOCO_GL setdefault above)
from robosuite.utils.input_utils import input2action  # noqa: E402


def run_teleop_episode(
    env,
    device,
    arm: str = "right",
    env_configuration: str = "single-arm-opposed",
    success_hold: int = 10,
) -> bool:
    """Drive one human-teleoperated episode via keyboard input.

    Structurally identical to ``collector.py``'s ``run_scripted_episode``
    (success-hold-count exit) — the only difference is the action SOURCE
    (human keyboard input via ``input2action`` vs. a computed waypoint).

    Args:
        env: a DataCollectionWrapper-wrapped SOARM env with
            ``has_renderer=True`` (built via ``build_recording_env``), NOT
            yet reset by the caller (this function owns reset, matching
            ``collect_human_trajectory``'s contract, since a human may need
            to retry a reset that raises transiently).
        device: a ``robosuite.devices.Keyboard`` instance already wired to
            the env's viewer keypress/keyup/keyrepeat callbacks.
        arm (str): which arm to control ("right" or "left").
        env_configuration (str): robosuite env configuration string passed
            through to ``input2action`` (SOARM is single-arm, so this only
            affects the IK_POSE-specific axis-swap branch inside
            ``input2action``, which OSC_POSE does not take).
        success_hold (int): consecutive ``_check_success()`` steps required
            to latch a win (matches ``collect_demonstration.py``'s 10).

    Returns:
        bool: True iff the episode completed successfully (success held for
            ``success_hold`` consecutive steps); False if the human triggered
            a device reset (early exit, nothing recorded should be kept).
    """
    reset_success = False
    while not reset_success:
        try:
            env.reset()
            reset_success = True
        except Exception:
            continue

    env.render()

    task_completion_hold_count = -1
    device.start_control()

    while True:
        # Set active robot (SOARM is single-arm, but mirror the reference's
        # bimanual-aware indexing so this stays correct if that ever changes).
        active_robot = env.robots[arm == "left"]

        action, grasp = input2action(
            device=device,
            robot=active_robot,
            active_arm=arm,
            env_configuration=env_configuration,
        )

        # A None action is the device's reset signal — bail out, nothing to
        # record (the caller's DataCollectionWrapper flush is simply not
        # flagged successful, so gather_demonstrations_as_hdf5 will drop it).
        if action is None:
            return False

        env.step(action)
        env.render()

        if task_completion_hold_count == 0:
            return True

        # robosuite success-hold state machine (collect_demonstration.py
        # lines 84-94): require `success_hold` consecutive successful steps
        # before latching a win.
        if env._check_success():
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1
            else:
                task_completion_hold_count = success_hold
        else:
            task_completion_hold_count = -1


def collect_teleop(
    bddl_file_name: str,
    hdf5_path: str,
    num_episodes: int = 1,
    tmp_directory: str = None,
    pos_sensitivity: float = 1.0,
    rot_sensitivity: float = 1.0,
    arm: str = "right",
    env_configuration: str = "single-arm-opposed",
) -> int:
    """Collect ``num_episodes`` human-teleoperated demos into one HDF5.

    Builds ONE on-screen recording env, wires a keyboard device to its
    viewer (no 3D-mouse input-device branch — D-02), loops
    ``run_teleop_episode`` for the requested number of attempts, then gathers
    the recorded npz episodes into a robomimic-schema HDF5 via the SAME
    ``gather_demonstrations_as_hdf5`` the scripted collector (04-02) calls.

    Args:
        bddl_file_name (str): Path to the task's BDDL file (fail-loudly
            validated by ``build_recording_env``).
        hdf5_path (str): Output HDF5 path.
        num_episodes (int): Number of teleop attempts to run (a human may
            trigger a device reset on some attempts without completing the
            task; only successful episodes are ever written).
        tmp_directory (str | None): DataCollectionWrapper scratch dir. When
            None, a FRESH unique dir is created per run under ``<hdf5 dir>/tmp/``
            so each run gathers only its own episodes (pass an explicit path to
            deliberately accumulate demos across runs).
        pos_sensitivity (float): Keyboard position input scale.
        rot_sensitivity (float): Keyboard rotation input scale.
        arm (str): which arm to control ("right" or "left").
        env_configuration (str): robosuite env configuration string.

    Returns:
        int: Number of demos actually written to the HDF5
            (``data.attrs['total']``).
    """
    from .raw_recorder import build_recording_env
    from .hdf5_writer import gather_demonstrations_as_hdf5

    assert os.path.exists(
        bddl_file_name
    ), f"[error] {bddl_file_name} does not exist!"

    os.makedirs(os.path.dirname(os.path.abspath(hdf5_path)), exist_ok=True)
    if tmp_directory is None:
        # FRESH scratch dir PER RUN. A fixed per-task path let successive runs
        # accumulate episodes in the same folder, and gather_demonstrations_as_hdf5
        # sweeps EVERY successful episode it finds — so a second run would silently
        # re-gather the first run's demos into the new HDF5 (observed: a "v2" file
        # came out holding the prior run's demo plus the new one). mkdtemp guarantees
        # each run gathers only its own episodes. Pass an explicit tmp_directory only
        # if you deliberately want to accumulate across runs.
        task_slug = os.path.basename(bddl_file_name).replace(".bddl", "")
        tmp_base = os.path.join(
            os.path.dirname(os.path.abspath(hdf5_path)), "tmp"
        )
        os.makedirs(tmp_base, exist_ok=True)
        tmp_directory = tempfile.mkdtemp(prefix=f"{task_slug}_teleop_", dir=tmp_base)

    # has_renderer=True: teleop requires a real on-screen viewer, unlike
    # collector.py's headless scripted collection. This installed robosuite
    # version (native `mujoco` bindings, not `mujoco_py`) always constructs
    # env.viewer as an OpenCVRenderer (see robosuite/environments/base.py),
    # not the older GLFW-based MujocoPyRenderer some reference scripts assume.
    env = build_recording_env(bddl_file_name, tmp_directory, has_renderer=True)

    # Keyboard's __init__ starts its own global pynput.keyboard.Listener
    # immediately (see robosuite/devices/keyboard.py) — it does NOT need (and
    # OpenCVRenderer does not support) per-key add_keypress_callback/
    # add_keyup_callback/add_keyrepeat_callback wiring through env.viewer;
    # that 3-callback pattern is specific to the older GLFW viewer.
    device = Keyboard(pos_sensitivity=pos_sensitivity, rot_sensitivity=rot_sensitivity)

    print(
        "\n[teleop] Keyboard controls (robosuite defaults — this project makes "
        "no changes to them):\n"
        "  Translation: 'w'/'a'/'s'/'d' (X-Y plane), 'r'/'f' (Z axis, up/down)\n"
        "  Rotation:    'z'/'x' (roll), 't'/'g' (pitch), 'c'/'v' (yaw)\n"
        "  Gripper:     SPACE (toggle open/close)\n"
        "  Reset:       'q' (aborts the current episode, nothing recorded)\n"
    )

    successes = 0
    try:
        for i in range(num_episodes):
            print(f"[teleop] Episode {i + 1}/{num_episodes} — drive the arm now.")
            if run_teleop_episode(
                env, device, arm=arm, env_configuration=env_configuration
            ):
                successes += 1
                print(f"[teleop] Episode {i + 1}: SUCCESS")
            else:
                print(f"[teleop] Episode {i + 1}: reset/incomplete, not recorded")
    finally:
        env.close()

    print(f"[collect_teleop] {successes} successful episode(s) recorded")

    written = gather_demonstrations_as_hdf5(tmp_directory, hdf5_path, bddl_file_name)
    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SOARM keyboard-only teleoperation collector (Phase 4, DATA-04)."
    )
    parser.add_argument("--bddl-file", type=str, required=True)
    parser.add_argument("--out", type=str, required=True, help="Output HDF5 path")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--pos-sensitivity", type=float, default=1.0)
    parser.add_argument("--rot-sensitivity", type=float, default=1.0)
    args = parser.parse_args()

    n = collect_teleop(
        args.bddl_file,
        args.out,
        num_episodes=args.episodes,
        pos_sensitivity=args.pos_sensitivity,
        rot_sensitivity=args.rot_sensitivity,
    )
    print(f"demos written: {n}")
