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
