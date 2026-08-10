"""Unit + integration tests for the spatial predicate classes (SPAT-05).

Unit tests (this file's ``TestLeftOfX``/``TestRightOfX``/``TestNearTo``/
``TestFarFrom`` classes) exercise ``LeftOfX``/``RightOfX``/``NearTo``/
``FarFrom`` directly against a stub object exposing only
``get_geom_state()["pos"]`` -- no BDDL/env construction needed (Task 1).

Integration tests (added in Tasks 2-3, name-filtered via pytest -k
"right_of"/"near"/"between") construct a real headless SOARM
``OffScreenRenderEnv`` against the new ``libero_spatial_soarm`` BDDL files
and prove goal-satisfaction across the full randomized init range (D-09),
directional correctness (predicates aren't trivially always-True), and --
for the new "between" 3rd-object region -- empirical collision-safety via
robosuite's sanctioned ``check_contact`` API (D-06, Pitfall 4).
"""

import os
import sys

os.environ.setdefault("MUJOCO_GL", "glfw")

# repo root = five levels up from this file's dir
# (envs -> libero -> libero -> LIBERO -> repo root)
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
# also anchor LIBERO/libero so `libero.*` resolves regardless of invocation dir
_LIBERO_LIBERO = os.path.join(_REPO_ROOT, "LIBERO", "libero")
if _LIBERO_LIBERO not in sys.path:
    sys.path.insert(0, _LIBERO_LIBERO)

import numpy as np
import pytest

from libero.envs.predicates.base_predicates import LeftOfX, RightOfX, NearTo, FarFrom
from libero.envs.predicates import eval_predicate_fn
from libero.envs import OffScreenRenderEnv

_BDDL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "bddl_files",
    "libero_spatial_soarm",
)
_RIGHT_OF_BDDL = os.path.normpath(
    os.path.join(_BDDL_DIR, "put_the_cream_cheese_to_the_right_of_the_bowl.bddl")
)
_NEAR_BDDL = os.path.normpath(
    os.path.join(_BDDL_DIR, "put_the_cream_cheese_near_the_bowl.bddl")
)
_BETWEEN_BDDL = os.path.normpath(
    os.path.join(
        _BDDL_DIR, "put_the_butter_between_the_bowl_and_the_cream_cheese.bddl"
    )
)

_N_RESETS = 20


def _build_env(bddl_path):
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        robots=["Soarm101"],
        camera_heights=128,
        camera_widths=128,
        has_renderer=False,
        has_offscreen_renderer=True,
    )
    return env


class _StubObj:
    """Minimal stand-in for ``ObjectState`` exposing only ``get_geom_state()``."""

    def __init__(self, pos):
        self._pos = np.array(pos, dtype=float)

    def get_geom_state(self):
        return {"pos": self._pos}


# ---------------------------------------------------------------------------
# Task 1: unit tests (no env/BDDL construction)
# ---------------------------------------------------------------------------

CLEAR_LEFT = (_StubObj([-0.20, 0.0, 0.0]), _StubObj([0.0, 0.0, 0.0]))  # arg1 left of arg2
CLEAR_RIGHT = (_StubObj([0.20, 0.0, 0.0]), _StubObj([0.0, 0.0, 0.0]))  # arg1 right of arg2
SAME_X = (_StubObj([0.0, 0.0, 0.0]), _StubObj([0.01, 0.0, 0.0]))  # within MARGIN
CLOSE_PAIR = (_StubObj([0.0, 0.0, 0.0]), _StubObj([0.05, 0.05, 0.0]))  # < THRESHOLD
FAR_PAIR = (_StubObj([0.0, 0.0, 0.0]), _StubObj([0.30, 0.30, 0.0]))  # > THRESHOLD


class TestLeftOfX:
    def test_clear_left_pair_true(self):
        arg1, arg2 = CLEAR_LEFT
        assert bool(LeftOfX()(arg1, arg2)) is True

    def test_clear_right_pair_false(self):
        arg1, arg2 = CLEAR_RIGHT
        assert bool(LeftOfX()(arg1, arg2)) is False

    def test_same_x_pair_false(self):
        arg1, arg2 = SAME_X
        assert bool(LeftOfX()(arg1, arg2)) is False


class TestRightOfX:
    def test_clear_right_pair_true(self):
        arg1, arg2 = CLEAR_RIGHT
        assert bool(RightOfX()(arg1, arg2)) is True

    def test_clear_left_pair_false(self):
        arg1, arg2 = CLEAR_LEFT
        assert bool(RightOfX()(arg1, arg2)) is False

    def test_same_x_pair_false(self):
        arg1, arg2 = SAME_X
        assert bool(RightOfX()(arg1, arg2)) is False


class TestNearTo:
    def test_close_pair_true(self):
        arg1, arg2 = CLOSE_PAIR
        assert bool(NearTo()(arg1, arg2)) is True

    def test_far_pair_false(self):
        arg1, arg2 = FAR_PAIR
        assert bool(NearTo()(arg1, arg2)) is False


class TestFarFrom:
    def test_far_pair_true(self):
        arg1, arg2 = FAR_PAIR
        assert bool(FarFrom()(arg1, arg2)) is True

    def test_close_pair_false(self):
        arg1, arg2 = CLOSE_PAIR
        assert bool(FarFrom()(arg1, arg2)) is False


# ---------------------------------------------------------------------------
# Task 2: integration tests -- right_of / near (reused, already-validated
# regions, D-06 fast path)
# ---------------------------------------------------------------------------


class TestRightOfTaskIntegration:
    def test_right_of_task_resets_satisfy_goal_20_of_20(self):
        env = _build_env(_RIGHT_OF_BDDL)
        try:
            successes = 0
            for _ in range(_N_RESETS):
                env.reset()
                if env.check_success():
                    successes += 1
            assert successes == _N_RESETS
        finally:
            env.close()

    def test_right_of_task_directional_correctness_reversed_args_false(self):
        env = _build_env(_RIGHT_OF_BDDL)
        try:
            env.reset()
            # Reversed argument order from the BDDL goal's actual order
            # (RightOfX cream_cheese_1 akita_black_bowl_1) -- proves the
            # predicate isn't trivially always-True regardless of order.
            result = eval_predicate_fn(
                "rightofx",
                env.env.object_states_dict["akita_black_bowl_1"],
                env.env.object_states_dict["cream_cheese_1"],
            )
            assert bool(result) is False
        finally:
            env.close()


class TestNearTaskIntegration:
    def test_near_task_resets_satisfy_goal_20_of_20(self):
        env = _build_env(_NEAR_BDDL)
        try:
            successes = 0
            for _ in range(_N_RESETS):
                env.reset()
                if env.check_success():
                    successes += 1
            assert successes == _N_RESETS
        finally:
            env.close()


# ---------------------------------------------------------------------------
# Task 3: integration tests -- "between" (new 3rd-object region, empirical
# collision-safety validation, D-06, Pitfall 4)
# ---------------------------------------------------------------------------


class TestBetweenTaskIntegration:
    def test_between_task_resets_satisfy_goal_20_of_20(self):
        env = _build_env(_BETWEEN_BDDL)
        try:
            successes = 0
            for _ in range(_N_RESETS):
                env.reset()
                if env.check_success():
                    successes += 1
            assert successes == _N_RESETS
        finally:
            env.close()

    def test_between_task_no_robot_butter_contact_at_rest_20_of_20(self):
        """D-06/Pitfall 4: the new butter_1 region needs its own empirical
        collision-safety check, not an assumption from coordinates alone.
        The arm should not be touching the new object merely by existing
        at its rest pose across the full randomized init range."""
        env = _build_env(_BETWEEN_BDDL)
        try:
            robot_model = env.env.robots[0].robot_model
            butter_model = env.env.get_object("butter_1")
            contact_count = 0
            for _ in range(_N_RESETS):
                env.reset()
                if env.env.check_contact(robot_model, butter_model):
                    contact_count += 1
            assert contact_count == 0
        finally:
            env.close()

    def test_between_task_directional_correctness_reversed_args_false(self):
        env = _build_env(_BETWEEN_BDDL)
        try:
            env.reset()
            # Reversed argument order from the BDDL goal's actual order
            # (RightOfX butter_1 akita_black_bowl_1) /
            # (LeftOfX butter_1 cream_cheese_1) -- proves neither predicate
            # is trivially always-True regardless of argument order.
            right_reversed = eval_predicate_fn(
                "rightofx",
                env.env.object_states_dict["akita_black_bowl_1"],
                env.env.object_states_dict["butter_1"],
            )
            left_reversed = eval_predicate_fn(
                "leftofx",
                env.env.object_states_dict["cream_cheese_1"],
                env.env.object_states_dict["butter_1"],
            )
            assert bool(right_reversed) is False
            assert bool(left_reversed) is False
        finally:
            env.close()
