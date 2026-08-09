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
