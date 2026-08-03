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
    it can actually grasp LIBERO objects (stock jaw was too small). Class name and
    the ``-1 => open, +1 => closed`` convention are preserved so the Phase-4
    collector / robot contract is unchanged.

    Args:
        idn (int or str): Number or some other unique identification string
            for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(os.path.join(ASSETS, "grippers/soarm_gripper.xml"), idn=idn)

    def format_action(self, action):
        """Maps the 1-D gripper action into the two jaw position targets.

        -1 => open, +1 => closed. Both jaw actuators share ctrlrange [0, 0.042]
        with current_action=-1 => 0 (closed) and +1 => 0.042 (open), so a CLOSE
        command (+1) drives both current elements toward -1 and an OPEN command
        (-1) drives them toward +1. Mirrors PandaGripper's two-element integrated
        action (here both elements move together — the jaws are symmetric).

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
        return np.array([0.065, 0.065])  # both jaws fully open (130 mm total; see xml deviation note)

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
