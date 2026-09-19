"""SOARM SO101 gripper model for robosuite 1.4 / LIBERO.

Mirrors robosuite's PandaGripper integrated-action pattern, collapsed to the
SO101's single jaw actuator. The gripper MJCF lives in the LIBERO fork's
asset tree, so the XML path is built from a fork-local ASSETS constant
instead of robosuite's path-completion helper.
"""
import os

import numpy as np

from robosuite.models.grippers.gripper_model import GripperModel

# Fork-local asset root (LIBERO/libero/libero/assets). Do NOT use robosuite's
# path-completion helper — it resolves inside robosuite site-packages.
ASSETS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "assets"))


class SoarmGripper(GripperModel):
    """SO-ARM100/101 parallel-jaw gripper (roboninecom upgrade).

    Two symmetric prismatic jaws (``gripper_left`` / ``gripper_right``) driven by
    one actuator each, collapsed to a single 1-D gripper action exactly like
    robosuite's ``PandaGripper``. 84 mm full opening stroke (+/-42 mm per jaw) so
    it can actually grasp LIBERO objects (stock jaw was too small).

    Class name is preserved so the Phase-4 collector / robot contract is
    unchanged; however TWIN-07 gap-closure (plan 260919-h8v) flipped the
    underlying MJCF joint axis/range polarity in ``soarm_gripper.xml`` so the
    sim's OPEN/CLOSE direction now agrees with the real SO-ARM101 hardware's
    positive-command-opens convention. The external action's sign meaning
    therefore flipped too: ``+1 => open, -1 => closed`` (was ``-1 => open,
    +1 => closed`` before the fix; see ``collector.py``'s ``OPEN_CMD``/
    ``CLOSE_CMD`` constants, which were updated to match). ``format_action``'s
    arithmetic below is unchanged; only the MJCF geometry mapping changed.

    Args:
        idn (int or str): Number or some other unique identification string
            for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(os.path.join(ASSETS, "grippers/soarm_gripper.xml"), idn=idn)

    def format_action(self, action):
        """Maps the 1-D gripper action into the two jaw position targets.

        External ``action``: +1 => open, -1 => closed (post-TWIN-07-fix
        convention; see class docstring). current_action's sign maps to
        physical position as: -1 => -0.042 (open), +1 => 0 (closed) — both
        jaw actuators share ctrlrange [-0.042, 0] (flipped from the pre-fix
        [0, 0.042] -- see soarm_gripper.xml). An OPEN command (external
        action=+1) drives both current_action elements toward -1; a CLOSE
        command (external action=-1) drives them toward +1. Mirrors
        PandaGripper's two-element integrated action (here both elements
        move together — the jaws are symmetric). The arithmetic below is
        unchanged by the fix; only the MJCF's joint-to-physical-position
        mapping changed.

        Args:
            action (np.array): gripper-specific action

        Raises:
            AssertionError: [Invalid action dimension size]
        """
        assert len(action) == self.dof
        self.current_action = np.clip(
            self.current_action + np.array([-1.0, -1.0]) * self.speed * np.sign(action),
            -1.0,
            1.0,
        )
        return self.current_action

    @property
    def init_qpos(self):
        return np.array([-0.042, -0.042])  # both jaws fully open (84 mm total; faithful roboninecom stroke; TWIN-07 fix flipped the range to [-0.042, 0])

    @property
    def speed(self):
        return 0.10  # jaw closes briskly within the FSM's GRASP_HOLD_STEPS budget

    @property
    def dof(self):
        return 1  # total action dim = OSC 6 + gripper 1 = 7 (Phase 3 contract)

    @property
    def _important_geoms(self):
        return {
            "left_finger": ["left_jaw_collision"],
            "right_finger": ["right_jaw_collision"],
            "left_fingerpad": ["left_jaw_collision"],
            "right_fingerpad": ["right_jaw_collision"],
        }
