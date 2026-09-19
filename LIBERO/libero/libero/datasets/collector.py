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

RETARGETED (2026-08-03, D-07/D-08): the original 3 frozen bowl->plate
``libero_spatial`` tasks are DROPPED (stock/faithful-84mm gripper can't grasp
the 11cm bowl, and the plate place-target is out of reach). This module now
targets ``libero_goal/put_the_cream_cheese_in_the_bowl.bddl`` — pick
``cream_cheese_1`` (sub-84mm), place into ``akita_black_bowl_1`` (goal:
``On(cream_cheese_1, akita_black_bowl_1)``). See ``TASK_BODY_MAP`` below for
the per-task pick/place body-name wiring the old 3-task set never needed
(all 3 old tasks happened to share identical body names).

Coordinate/tuning notes (empirically measured against the real SOARM env for
the retargeted cream_cheese/bowl task; table top at z=0.90, confirmed accurate
and unchanged from the original bowl investigation):
  * ``cream_cheese_1`` rests FLAT on the table (it does not tip — the asset is
    already authored lying on its largest face): body origin settles at
    z ~= 0.909, i.e. object top surface ~= 0.918 (half-thickness ~0.009 m; the
    object's long/medium faces, ~8.1x4.3 cm, lie horizontal). eef starts at
    z ~= 1.08 (not 1.13 — that was the old bowl task's start height).
  * ``akita_black_bowl_1`` (place target, used passively/goal-only here) rests
    at body origin z ~= 0.898 (same object/geometry as the old bowl tasks).
  * RESOLVED (2026-08-03, 3rd attempt): a previously-reported "vertical
    reach-depth" blocker (eef asymptoting ~1-3cm above the object regardless
    of GRASP_Z_OFFSET/controller-type) was a MISDIAGNOSIS, corrected the same
    day — direct MuJoCo contact inspection showed the real cause was the old
    ``akita_black_bowl_region`` sitting almost dead-ahead of the base (y~=0),
    directly in the forearm's natural sweep corridor, causing genuine
    jaw/forearm-vs-bowl collisions (measured actuator_force never exceeded
    ~30% of the +/-2.94Nm budget — not a torque ceiling). A further empirical
    XY sweep (this 3rd attempt) additionally found the arm's reachable XY
    envelope, while holding a FIXED end-effector orientation (this FSM never
    commands a rotation delta), is tightly bounded to roughly |y| <~ 0.04 m
    from the base's centerline REGARDLESS of x/radial distance — a genuine
    kinematic constraint of this 5-DOF arm at a fixed orientation, not a bug.
    Fix applied: stripped unused wine_bottle/cabinet/stove/wine_rack clutter
    from the BDDL and repositioned both ``akita_black_bowl_region`` and
    ``cream_cheese_region`` to sit within that reachable |y| band at
    different radii (angularly/radially separated so the arm's sweep to one
    object doesn't pass through the other), see the BDDL file itself for the
    exact ranges. Validated via the real ``collect_task`` path with direct
    contact inspection: zero jaw/forearm-vs-bowl collisions across 20
    episodes (only the expected, intentional cream_cheese-vs-bowl contact
    during place_descend/release), 16/20 and 8/9 successes in two
    independent batches. See 04-02-SUMMARY.md's dated sections for the full
    measurement trail across all 3 attempts.
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
OPEN_CMD = 1.0           # gripper action element: open jaw (post-Phase-10-TWIN-07-fix polarity: +1=open, -1=closed)
CLOSE_CMD = -1.0         # gripper action element: close jaw
HOVER_HEIGHT = 0.12      # m above object/place-target for approach/lift/transport/retreat
GRASP_Z_OFFSET = 0.015   # m above the pick object's body origin the jaw descends to
PLACE_Z_OFFSET = 0.06    # m above the place target the pick object is released from
KP_POS = 25.0            # proportional gain: metric error (m) -> normalized action
XY_TOL = 0.020           # m horizontal tolerance for phase transitions
Z_TOL = 0.010            # m vertical tolerance for phase transitions (tightened
                         # 2026-08-03, was 0.025: the old value let descend->
                         # grasp transition ~1.5cm above the true grasp point
                         # [Rule 1 bug — a loose Z_TOL, not a hardware limit],
                         # so the jaw only nudged the object instead of
                         # enclosing it; see collector.py module docstring)
GRASP_HOLD_STEPS = 20    # extra steps held at the object so the jaw finishes closing
RELEASE_HOLD_STEPS = 10  # extra steps held above the place target so the jaw finishes opening

# Phases in which the gripper is commanded CLOSED (holding the picked object).
_CLOSED_PHASES = ("grasp", "lift", "transport", "place_descend")

# RETARGETED (D-07/D-08, 2026-08-03): the original 3-task frozen bowl->plate
# libero_spatial list is DROPPED (11cm bowl > 84mm faithful jaw; plate
# place-target also out of the ~0.479m arm reach). Single sub-84mm in-reach
# task: pick cream_cheese_1 (narrow face well under the 84mm jaw), place into
# akita_black_bowl_1 (goal: On(cream_cheese_1, akita_black_bowl_1)). Both the
# object-init region (~0.355m from base) and the place region (~0.29m from
# base) sit inside the arm's 0.479m max reach -- reuses the same "table"
# fixture/base offset already validated by the old 3 tasks.
TASKS = [
    "put_the_cream_cheese_in_the_bowl.bddl",
]

# Per-task pick/place MuJoCo body-name wiring. The old 3-task list never
# needed this (all 3 shared identical body names: akita_black_bowl_1 /
# plate_1), but the retargeted task uses different names for both roles, so
# collect_task/collect_all must thread the correct pair through explicitly
# rather than relying on run_scripted_episode's hardcoded defaults.
TASK_BODY_MAP = {
    "put_the_cream_cheese_in_the_bowl.bddl": {
        "bowl_body": "cream_cheese_1",       # pick object (sub-84mm)
        "plate_body": "akita_black_bowl_1",  # place target (goal container)
    },
}

# This file lives at .../libero/libero/datasets/collector.py, so ".." resolves
# to .../libero/libero, the sibling of bddl_files. (This project's convention
# deliberately avoids get_libero_path — the local ~/.libero/config.yaml is stale
# and points outside this repo; see soarm_sanity.py's hardcoded BDDL_DIR.)
# RETARGETED: libero_goal (was libero_spatial) -- put_the_cream_cheese_in_the_bowl
# lives in the libero_goal task suite.
BDDL_DIR = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "bddl_files",
        "libero_goal",
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
    bowl_body="cream_cheese_1",
    plate_body="akita_black_bowl_1",
    max_steps=300,
    success_hold=10,
):
    """Drive one scripted episode through the waypoint FSM on a real env.

    Reset is the CALLER's responsibility (mirrors ``vla/eval_loop.run_episode``'s
    contract inverted — here ``collect_task`` resets in a loop so it can count
    attempts). Reads ground-truth pick/place-target positions off the env each
    step, feeds them + the live eef position to ``compute_waypoint_action``,
    steps the env, and tracks a robosuite-style ``task_completion_hold_count``
    (success must hold for ``success_hold`` consecutive steps) as the primary
    exit condition.

    Args:
        env: a DataCollectionWrapper-wrapped SOARM env, freshly ``reset()`` by
            the caller. ``obj_body_id`` / ``sim`` / ``_check_success`` /
            ``_get_observations`` are reachable through robosuite's Wrapper proxy.
        bowl_body (str): MuJoCo body name of the PICK object (param name kept
            from the original bowl->plate tasks; defaults to the retargeted
            task's pick object, ``cream_cheese_1`` — see ``TASK_BODY_MAP``).
        plate_body (str): MuJoCo body name of the PLACE target (param name kept
            from the original bowl->plate tasks; defaults to the retargeted
            task's place target, ``akita_black_bowl_1``).
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
    bowl_body="cream_cheese_1",
    plate_body="akita_black_bowl_1",
    target_successes=120,
    max_attempts=300,
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
        bowl_body (str): MuJoCo body name of the PICK object, threaded through to
            ``run_scripted_episode`` (param name kept from the original bowl->plate
            tasks). Defaults to the retargeted task's pick object; pass the
            ``TASK_BODY_MAP`` entry explicitly for a given ``bddl_file_name``
            rather than relying on this default when adding more tasks.
        plate_body (str): MuJoCo body name of the PLACE target, threaded through
            to ``run_scripted_episode`` (param name kept from the original
            bowl->plate tasks). Defaults to the retargeted task's place target.
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
            if run_scripted_episode(env, bowl_body=bowl_body, plate_body=plate_body):
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


def collect_all(target_per_task=120, output_dir=None, max_attempts_per_task=300):
    """Collect the full (now single-task) scripted dataset — DATA-01's primary output.

    RETARGETED (D-07/D-08): this used to loop over 3 frozen bowl->plate tasks
    at ``target_per_task=40`` each (40x3=120 total, a 20% buffer over the 100+
    requirement). With only 1 task now, ``target_per_task`` must absorb that
    entire buffer itself (default raised to 120) so the "100+" requirement is
    still met from a single task's HDF5 rather than silently undershooting it.

    Args:
        target_per_task (int): Success target for the (single) task. Default 120
            preserves the original 3-task total/buffer now that there is only
            one task to collect from.
        output_dir (str | None): Where the per-task HDF5s land. Defaults to
            ``LIBERO/libero/datasets/soarm_spatial`` — NOT ``get_libero_path``
            (the local ~/.libero/config.yaml is stale; this project's convention
            avoids it, see soarm_sanity.py).
        max_attempts_per_task (int): Per-task hard attempt cap.

    Returns:
        dict: {task_slug: demos_written} for the (single) retargeted task.
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
        body_names = TASK_BODY_MAP.get(task, {})
        count = collect_task(
            os.path.join(BDDL_DIR, task),
            hdf5_path,
            bowl_body=body_names.get("bowl_body", "cream_cheese_1"),
            plate_body=body_names.get("plate_body", "akita_black_bowl_1"),
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
    parser.add_argument("--target-per-task", type=int, default=120)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-attempts-per-task", type=int, default=300)
    args = parser.parse_args()

    out = collect_all(
        target_per_task=args.target_per_task,
        output_dir=args.output_dir,
        max_attempts_per_task=args.max_attempts_per_task,
    )
    print(out)
