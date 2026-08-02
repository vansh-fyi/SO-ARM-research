"""Hand-coded waypoint scripted demonstration collector for SOARM (Phase 4, D-01).

Research (04-RESEARCH.md) confirmed LIBERO/robosuite ship no built-in scripted
expert policy for pick-place, and VLA-rollout filtering is non-viable (OFT/pi0
sit at 0% zero-shot success on SOARM per Phase 3). The resolved approach is a
pure, hand-coded waypoint state machine driven by GROUND-TRUTH object positions
(``sim.data.body_xpos[obj_body_id[...]]``) — not learned perception.

This module has two layers:

  * ``compute_waypoint_action`` — a PURE 8-phase FSM step function (no env/sim
    access, only numpy math). Independently unit-tested in test_collector.py.
  * ``run_scripted_episode`` / ``collect_task`` / ``collect_all`` — the driver
    that reads ground-truth positions off a real env, feeds them to the FSM,
    steps the env, gates on success, and writes robomimic-schema HDF5 via
    04-01's ``gather_demonstrations_as_hdf5``.

The 8 phases: approach -> descend -> grasp -> lift -> transport -> place_descend
-> release -> retreat -> done.

Coordinate/tuning notes (empirically measured against the real SOARM env, per
RESEARCH.md Open Question 1 — these are the tuned starting values, iterate here
if a task's success rate is low):
  * eef starts high (z~1.13); bowls sit on the table at z~0.97 (the ramekin
    task's bowl at z~1.08); plates at z~0.97.
  * Actions are OSC_POSE deltas in the controller's normalized [-1, 1] range
    (input_max/min), NOT raw metric offsets — so we apply a proportional gain
    ``KP_POS`` to the metric position error and clip to [-1, 1]. A ~0.05 m error
    saturates the command, giving brisk-but-stable motion within the step budget.
  * Gripper convention (SoarmGripper.format_action): -1 => open, +1 => closed.
"""

import argparse
import os

import numpy as np

# Set the MuJoCo GL backend before any import that triggers MuJoCo (mirrors
# raw_recorder.py / hdf5_writer.py). setdefault so a Linux osmesa value stands.
os.environ.setdefault("MUJOCO_GL", "glfw")

# --- FSM tuning constants (empirically tuned starting values) ---------------
OPEN_CMD = -1.0          # gripper action element: open jaw
CLOSE_CMD = 1.0          # gripper action element: close jaw
HOVER_HEIGHT = 0.12      # m above bowl/plate for approach/lift/transport/retreat
GRASP_Z_OFFSET = 0.015   # m above the bowl body origin the jaw descends to
PLACE_Z_OFFSET = 0.06    # m above the plate the bowl is released from
KP_POS = 25.0            # proportional gain: metric error (m) -> normalized action
XY_TOL = 0.020           # m horizontal tolerance for phase transitions
Z_TOL = 0.025            # m vertical tolerance for phase transitions
GRASP_HOLD_STEPS = 20    # extra steps held at the bowl so the jaw finishes closing
RELEASE_HOLD_STEPS = 10  # extra steps held above the plate so the jaw finishes opening

# Phases in which the gripper is commanded CLOSED (holding the bowl).
_CLOSED_PHASES = ("grasp", "lift", "transport", "place_descend")

# Corrected 3-task frozen list — copied verbatim from
# explorations/soarm_sanity.py lines 65-69. The 02-01 candidate
# ``next_to_the_plate`` was swapped out (bowl region 0.498 m > SOARM's 0.479 m
# reach); do NOT reintroduce it.
TASKS = [
    "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate.bddl",
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bddl",
    "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate.bddl",
]

# This file lives at .../libero/libero/datasets/collector.py, so ".." resolves
# to .../libero/libero, the sibling of bddl_files. (This project's convention
# deliberately avoids get_libero_path — the local ~/.libero/config.yaml is stale
# and points outside this repo; see soarm_sanity.py's hardcoded BDDL_DIR.)
BDDL_DIR = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "bddl_files",
        "libero_spatial",
    )
)


def _reached(target, eef_pos):
    """True if eef is within (XY_TOL, Z_TOL) of a 3D target point."""
    d = np.asarray(target, dtype=float) - np.asarray(eef_pos, dtype=float)
    return float(np.linalg.norm(d[:2])) < XY_TOL and abs(float(d[2])) < Z_TOL


def _pose_action(target, eef_pos, gripper_cmd):
    """Build a 7-D OSC_POSE action toward ``target`` with a given gripper cmd.

    Proportional position controller: normalized delta = clip(KP * error, -1, 1).
    Rotation deltas are always zero (top-down grasp, no reorientation).
    """
    err = np.asarray(target, dtype=float) - np.asarray(eef_pos, dtype=float)
    dpos = np.clip(KP_POS * err, -1.0, 1.0)
    return np.array(
        [dpos[0], dpos[1], dpos[2], 0.0, 0.0, 0.0, gripper_cmd], dtype=float
    )


def compute_waypoint_action(phase, eef_pos, bowl_pos, plate_pos, gripper_closed):
    """Pure 8-phase FSM step: one waypoint action from ground-truth positions.

    Args:
        phase (str): Current FSM phase (one of the 8 phases + "done").
        eef_pos (np.ndarray): Current end-effector world position (3,).
        bowl_pos (np.ndarray): Ground-truth bowl world position (3,).
        plate_pos (np.ndarray): Ground-truth plate world position (3,).
        gripper_closed (bool): Whether the close command has been committed
            (tracked explicitly so this function stays pure — no hidden state).

    Returns:
        tuple[np.ndarray, str, bool]: (7-D action, next_phase, gripper_closed).
    """
    bowl_pos = np.asarray(bowl_pos, dtype=float)
    plate_pos = np.asarray(plate_pos, dtype=float)
    eef_pos = np.asarray(eef_pos, dtype=float)

    hover_bowl = bowl_pos + np.array([0.0, 0.0, HOVER_HEIGHT])
    grasp_pt = bowl_pos + np.array([0.0, 0.0, GRASP_Z_OFFSET])
    hover_plate = plate_pos + np.array([0.0, 0.0, HOVER_HEIGHT])
    place_pt = plate_pos + np.array([0.0, 0.0, PLACE_Z_OFFSET])

    # Determine target + next_phase per current phase.
    if phase == "approach":
        target = hover_bowl
        next_phase = "descend" if _reached(target, eef_pos) else "approach"
    elif phase == "descend":
        target = grasp_pt
        next_phase = "grasp" if _reached(target, eef_pos) else "descend"
    elif phase == "grasp":
        target = grasp_pt
        # Hold at the bowl and commit the close; advance only once committed.
        next_phase = "lift" if gripper_closed else "grasp"
        gripper_closed = True
    elif phase == "lift":
        target = hover_bowl
        next_phase = "transport" if _reached(target, eef_pos) else "lift"
    elif phase == "transport":
        target = hover_plate
        next_phase = "place_descend" if _reached(target, eef_pos) else "transport"
    elif phase == "place_descend":
        target = place_pt
        next_phase = "release" if _reached(target, eef_pos) else "place_descend"
    elif phase == "release":
        target = place_pt
        # Hold above the plate and commit the open; advance only once released.
        next_phase = "retreat" if not gripper_closed else "release"
        gripper_closed = False
    elif phase == "retreat":
        target = hover_plate
        next_phase = "done" if _reached(target, eef_pos) else "retreat"
    else:  # "done" (or any unknown phase): hold, gripper open
        target = eef_pos
        next_phase = "done"

    # Gripper command follows the RESULTING phase, so transition steps already
    # carry the correct command (e.g. the descend->grasp step commands close).
    gripper_cmd = CLOSE_CMD if next_phase in _CLOSED_PHASES else OPEN_CMD
    action = _pose_action(target, eef_pos, gripper_cmd)
    return action, next_phase, gripper_closed


def run_scripted_episode(
    env,
    bowl_body="akita_black_bowl_1",
    plate_body="plate_1",
    max_steps=300,
    success_hold=10,
):
    """Drive one scripted episode through the waypoint FSM on a real env.

    Reset is the CALLER's responsibility (mirrors ``vla/eval_loop.run_episode``'s
    contract inverted — here ``collect_task`` resets in a loop so it can count
    attempts). Reads ground-truth bowl/plate positions off the env each step,
    feeds them + the live eef position to ``compute_waypoint_action``, steps the
    env, and tracks a robosuite-style ``task_completion_hold_count`` (success must
    hold for ``success_hold`` consecutive steps) as the primary exit condition.

    Args:
        env: a DataCollectionWrapper-wrapped SOARM env, freshly ``reset()`` by
            the caller. ``obj_body_id`` / ``sim`` / ``_check_success`` /
            ``_get_observations`` are reachable through robosuite's Wrapper proxy.
        bowl_body (str): MuJoCo body name of the bowl to pick.
        plate_body (str): MuJoCo body name of the target plate.
        max_steps (int): hard per-episode step cap (fail-safe against a stuck FSM).
        success_hold (int): consecutive ``_check_success()`` steps that latch a win.

    Returns:
        bool: True iff success held for ``success_hold`` consecutive steps.
    """
    # Initial obs after the caller's reset (Wrapper proxies to the base env).
    obs = env._get_observations()

    phase = "approach"
    gripper_closed = False
    grasp_hold = 0
    release_hold = 0
    task_completion_hold_count = -1

    # Frozen bowl reference: once the bowl is grasped it rises WITH the eef, so a
    # live bowl+HOVER lift target would chase the eef forever. Track the bowl
    # only through the grasp; freeze it for lift onward.
    bowl_ref = np.array(env.sim.data.body_xpos[env.obj_body_id[bowl_body]])

    for _ in range(max_steps):
        eef_pos = np.array(obs["robot0_eef_pos"])
        if phase in ("approach", "descend", "grasp"):
            bowl_ref = np.array(env.sim.data.body_xpos[env.obj_body_id[bowl_body]])
        plate_pos = np.array(env.sim.data.body_xpos[env.obj_body_id[plate_body]])

        action, next_phase, gripper_closed = compute_waypoint_action(
            phase, eef_pos, bowl_ref, plate_pos, gripper_closed
        )

        # Hold grasp/release a few physics steps so the 1-DOF jaw (speed 0.10/step)
        # finishes moving before the arm advances — otherwise the bowl slips.
        if phase == "grasp" and next_phase == "lift" and grasp_hold < GRASP_HOLD_STEPS:
            grasp_hold += 1
            next_phase = "grasp"
            gripper_closed = True
        if (
            phase == "release"
            and next_phase == "retreat"
            and release_hold < RELEASE_HOLD_STEPS
        ):
            release_hold += 1
            next_phase = "release"
            gripper_closed = False

        obs, _, _, _ = env.step(action)
        phase = next_phase

        # robosuite success-hold state machine (collect_demonstration.py lines 84-94):
        # require `success_hold` consecutive successful steps before latching a win.
        if task_completion_hold_count == 0:
            return True
        if env._check_success():
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1
            else:
                task_completion_hold_count = success_hold
        else:
            task_completion_hold_count = -1

    return False


def collect_task(
    bddl_file_name,
    hdf5_path,
    target_successes=40,
    max_attempts=200,
    tmp_directory=None,
):
    """Collect ``target_successes`` scripted demos for one task -> one HDF5.

    Builds ONE headless recording env (reused across all attempts — cheaper than
    rebuilding per attempt), loops reset + ``run_scripted_episode`` until the
    success target is met or ``max_attempts`` is exhausted, then gathers the
    recorded npz episodes into a robomimic-schema HDF5 exactly once.

    Args:
        bddl_file_name (str): Path to the task's BDDL file (fail-loudly validated).
        hdf5_path (str): Output HDF5 path for this task's demos.
        target_successes (int): Stop once this many episodes succeed.
        max_attempts (int): Hard cap on reset attempts (T-04-02-01: prevents an
            unreachable target from looping forever — surfaces as a low return).
        tmp_directory (str | None): DataCollectionWrapper scratch dir; defaults to
            ``<hdf5 dir>/tmp/<task_slug>``.

    Returns:
        int: Number of demos actually written to the HDF5 (``data.attrs['total']``).
    """
    from .raw_recorder import build_recording_env
    from .hdf5_writer import gather_demonstrations_as_hdf5

    assert os.path.exists(
        bddl_file_name
    ), f"[error] {bddl_file_name} does not exist!"

    if tmp_directory is None:
        task_slug = os.path.basename(bddl_file_name).replace(".bddl", "")
        tmp_directory = os.path.join(
            os.path.dirname(os.path.abspath(hdf5_path)), "tmp", task_slug
        )
    os.makedirs(os.path.dirname(os.path.abspath(hdf5_path)), exist_ok=True)

    env = build_recording_env(bddl_file_name, tmp_directory, has_renderer=False)

    successes = 0
    attempts = 0
    try:
        while successes < target_successes and attempts < max_attempts:
            env.reset()
            attempts += 1
            if run_scripted_episode(env):
                successes += 1
    finally:
        env.close()

    print(
        f"[collect_task] {os.path.basename(bddl_file_name)}: "
        f"{successes} successes / {attempts} attempts"
    )

    # Gather ALL recorded episodes; gather re-derives success per episode from the
    # OR-accumulated npz flag (intentionally redundant with the attempt count).
    written = gather_demonstrations_as_hdf5(tmp_directory, hdf5_path, bddl_file_name)
    return written


def collect_all(target_per_task=40, output_dir=None, max_attempts_per_task=200):
    """Collect the full 3-task scripted dataset (the phase's primary DATA-01 output).

    Args:
        target_per_task (int): Success target per task (D-04: ~40/task, 120 total,
            a 20% buffer over the 100+ requirement).
        output_dir (str | None): Where the per-task HDF5s land. Defaults to
            ``LIBERO/libero/datasets/soarm_spatial`` — NOT ``get_libero_path``
            (the local ~/.libero/config.yaml is stale; this project's convention
            avoids it, see soarm_sanity.py).
        max_attempts_per_task (int): Per-task hard attempt cap.

    Returns:
        dict: {task_slug: demos_written} for the 3 frozen tasks.
    """
    if output_dir is None:
        output_dir = os.path.normpath(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..",
                "..",
                "datasets",
                "soarm_spatial",
            )
        )
    os.makedirs(output_dir, exist_ok=True)

    results = {}
    for task in TASKS:
        task_slug = task.replace(".bddl", "")
        hdf5_path = os.path.join(output_dir, f"{task_slug}_demo.hdf5")
        tmp_directory = os.path.join(output_dir, "tmp", task_slug)
        count = collect_task(
            os.path.join(BDDL_DIR, task),
            hdf5_path,
            target_successes=target_per_task,
            max_attempts=max_attempts_per_task,
            tmp_directory=tmp_directory,
        )
        results[task_slug] = count
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SOARM scripted waypoint demo collector (Phase 4, DATA-01)."
    )
    parser.add_argument("--target-per-task", type=int, default=40)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-attempts-per-task", type=int, default=200)
    args = parser.parse_args()

    out = collect_all(
        target_per_task=args.target_per_task,
        output_dir=args.output_dir,
        max_attempts_per_task=args.max_attempts_per_task,
    )
    print(out)
