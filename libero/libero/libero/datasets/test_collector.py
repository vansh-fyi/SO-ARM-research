"""Tests for collector.py — the scripted waypoint SOARM demo collector (Phase 4).

Two tiers, matching 04-RESEARCH.md's Validation Architecture:

  * ``test_waypoint_*`` — pure, no-sim unit tests of ``compute_waypoint_action``,
    the 8-phase FSM step function. Hand-picked eef/bowl/plate numpy positions;
    no MockEnv needed (the function takes plain positions, not an env). Mirrors
    ``vla/test_eval_loop.py``'s mock/synthetic-input style.
  * ``test_run_scripted_episode_reaches_success_within_budget`` — a REAL
    (no-mock) local-sim integration smoke test exercising ``collect_task``
    against TASKS[0] end-to-end (build env -> scripted rollout -> HDF5 write),
    asserting >=2 successful demos within a small attempt budget.

Import-path note (mirrors vla/test_eval_loop.py / test_hdf5_writer.py): insert
the repo root so ``LIBERO.libero.libero.envs`` absolute imports resolve, and
``LIBERO/libero`` so ``libero.datasets.*`` resolves regardless of invocation dir.
"""

import os
import sys

os.environ.setdefault("MUJOCO_GL", "glfw")

# repo root = four levels up (datasets -> libero -> libero -> LIBERO -> root)
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_LIBERO_LIBERO = os.path.join(_REPO_ROOT, "LIBERO", "libero")
if _LIBERO_LIBERO not in sys.path:
    sys.path.insert(0, _LIBERO_LIBERO)

import numpy as np
import pytest

from libero.datasets.collector import (
    compute_waypoint_action,
    HOVER_HEIGHT,
    CLOSE_CMD,
    OPEN_CMD,
    TASKS,
    BDDL_DIR,
)

# Reusable hand-picked positions (world frame, matching the real env geometry:
# bowls sit on the table at z~0.97, plate at z~0.97, eef starts high at z~1.13).
BOWL = np.array([-0.075, 0.012, 0.97])
PLATE = np.array([0.047, 0.198, 0.97])


def _unpack(phase, eef, gripper_closed):
    action, next_phase, gc = compute_waypoint_action(
        phase, np.asarray(eef, dtype=float), BOWL, PLATE, gripper_closed
    )
    return np.asarray(action), next_phase, gc


def test_waypoint_action_is_7d_vector():
    action, _, _ = _unpack("approach", BOWL + [0.3, 0.3, 0.4], False)
    assert action.shape == (7,)
    # rotation deltas are always zero (top-down grasp, no reorientation)
    assert np.allclose(action[3:6], 0.0)


def test_waypoint_approach_far_points_toward_bowl_and_stays_open():
    # eef far above + offset in +x/+y from the bowl; hover target is
    # bowl + [0,0,HOVER], which is BELOW this eef.
    eef = BOWL + np.array([0.30, 0.25, 0.40])
    action, next_phase, _ = _unpack("approach", eef, False)
    # dx/dy point back toward the bowl (negative, since eef is +x/+y of bowl)
    assert action[0] < 0
    assert action[1] < 0
    # eef is above the hover height -> descend -> dz negative
    assert action[2] < 0
    # gripper open (<= 0) before grasp
    assert action[6] <= 0
    # not close enough yet -> phase unchanged
    assert next_phase == "approach"


def test_waypoint_approach_at_hover_transitions_to_descend():
    eef = BOWL + np.array([0.0, 0.0, HOVER_HEIGHT])
    _, next_phase, _ = _unpack("approach", eef, False)
    assert next_phase == "descend"


def test_waypoint_descend_at_bowl_transitions_to_grasp_and_closes():
    eef = BOWL.copy()  # at bowl height
    action, next_phase, _ = _unpack("descend", eef, False)
    assert next_phase == "grasp"
    # gripper element goes positive (close command) on the descend->grasp step
    assert action[6] > 0


def test_waypoint_grasp_transitions_to_lift_after_close_committed():
    # First grasp step with gripper still open -> stays in grasp, commits close.
    action0, next0, gc0 = _unpack("grasp", BOWL.copy(), False)
    assert next0 == "grasp"
    assert action0[6] > 0
    assert gc0 is True
    # Once the close has been committed (gripper_closed=True) -> transition to lift.
    _, next1, _ = _unpack("grasp", BOWL.copy(), True)
    assert next1 == "lift"


def test_waypoint_gripper_positive_from_grasp_through_place_descend():
    for phase in ("grasp", "lift", "transport", "place_descend"):
        action, _, _ = _unpack(phase, BOWL.copy(), True)
        assert action[6] > 0, f"gripper should be closed (positive) in {phase}"


def test_waypoint_gripper_negative_from_release_onward():
    # release still holding (gripper_closed=True) issues the OPEN command
    action, _, gc = _unpack("release", PLATE.copy(), True)
    assert action[6] < 0
    assert gc is False
    # retreat is open
    action2, _, _ = _unpack("retreat", PLATE + [0, 0, HOVER_HEIGHT], False)
    assert action2[6] < 0


def test_waypoint_retreat_above_plate_transitions_to_done():
    eef = PLATE + np.array([0.0, 0.0, HOVER_HEIGHT])
    _, next_phase, _ = _unpack("retreat", eef, False)
    assert next_phase == "done"


def test_tasks_constant_is_corrected_three_task_list():
    assert len(TASKS) == 3
    joined = " ".join(TASKS)
    assert "on_the_ramekin_and_place_it_on_the_plate" in joined
    assert "table_center" in joined
    assert "between_the_plate_and_the_ramekin" in joined
    # The stale, out-of-reach task must NOT be present.
    assert "next_to_the_plate" not in joined


# ----------------------------------------------------------------------------
# Real (no-mock) local-sim integration tests — Task 2 acceptance.
#
# NOTE (blocker, see 04-02-SUMMARY.md "Blocker"): the SOARM gripper cannot
# grasp/lift the akita_black_bowl — the arm's vertical reach bottoms out ~0.02 m
# ABOVE the settled bowl rim, and the jaw opening (~0.03 m) is far smaller than
# the bowl (~0.09 m). So no scripted (or learned — Phase 3 = 0%) policy reaches
# On(bowl, plate) with the current robot. The first test proves the collector
# PLUMBING is correct end-to-end (runs a real env, records, writes a valid HDF5);
# the second encodes the TARGET behavior and is xfail'd against that blocker,
# with the >=2 assertion preserved (NOT weakened) so it flips to xpass the day a
# gripper/reach redesign makes grasping feasible.
# ----------------------------------------------------------------------------


def test_collect_task_runs_end_to_end_and_writes_valid_hdf5(tmp_path):
    """The collector drives a real SOARM env reset->rollout->HDF5 write without
    error and emits a schema-valid HDF5 (proves DATA-01 plumbing independent of
    whether the scripted policy actually completes the task)."""
    import h5py
    from libero.datasets.collector import collect_task

    hdf5_path = str(tmp_path / "smoke_demo.hdf5")
    tmp_dir = str(tmp_path / "raw")
    count = collect_task(
        os.path.join(BDDL_DIR, TASKS[0]),
        hdf5_path=hdf5_path,
        target_successes=1,
        max_attempts=2,
        tmp_directory=tmp_dir,
    )
    assert isinstance(count, int) and count >= 0
    assert os.path.exists(hdf5_path)
    with h5py.File(hdf5_path, "r") as f:
        # robomimic-schema top-level group + total attr always present.
        assert "data" in f
        assert "total" in f["data"].attrs
        assert int(f["data"].attrs["total"]) == count


@pytest.mark.xfail(
    strict=False,
    reason=(
        "BLOCKER: SOARM gripper cannot grasp/lift the akita_black_bowl (vertical "
        "reach ~0.02m short of the settled bowl rim; jaw opening ~0.03m << bowl "
        "~0.09m). No scripted trajectory reaches On(bowl,plate). See 04-02-SUMMARY.md."
    ),
)
def test_run_scripted_episode_reaches_success_within_budget(tmp_path):
    from libero.datasets.collector import collect_task

    hdf5_path = str(tmp_path / "smoke_demo.hdf5")
    tmp_dir = str(tmp_path / "raw")
    count = collect_task(
        os.path.join(BDDL_DIR, TASKS[0]),
        hdf5_path=hdf5_path,
        target_successes=2,
        max_attempts=50,
        tmp_directory=tmp_dir,
    )
    assert count >= 2, f"scripted collector reached only {count} successes on TASKS[0]"
